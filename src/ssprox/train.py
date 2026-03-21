import pickle
import time
from contextlib import nullcontext
from itertools import cycle
from pathlib import Path
import math 

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from ssprox.model import GPT


def resolve_dtype(dtype: str) -> str:
    if dtype != "auto":
        return dtype
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bfloat16"
    if torch.cuda.is_available():
        return "float16"
    return "float32"


def build_autocast(device_type: str, dtype: str):
    if device_type == "cpu" or dtype == "float32":
        return nullcontext()
    ptdtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    return torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def load_meta(data_dir: Path) -> dict:
    meta_path = data_dir / "meta.pkl"
    if not meta_path.exists():
        return {}
    with meta_path.open("rb") as handle:
        return pickle.load(handle)


def validate_dataset(data_dir: Path) -> None:
    expected = [data_dir / "train.bin", data_dir / "val.bin"]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset binaries are missing. Expected files: "
            f"{', '.join(missing)}. Run the corresponding prepare.py first.",
        )


class NextTokenDataset(Dataset):
    def __init__(self, path: Path, block_size: int) -> None:
        file_size = path.stat().st_size
        element_size = torch.tensor([], dtype=torch.uint16).element_size()
        token_count = file_size // element_size
        self.tokens = torch.from_file(
            str(path),
            shared=False,
            size=token_count,
            dtype=torch.uint16,
        )
        self.block_size = block_size
        if self.tokens.numel() <= block_size:
            raise ValueError(
                f"{path.name} is too small for block_size={block_size}. "
                "Prepare a larger dataset or lower the block size.",
            )

    def __len__(self) -> int:
        return self.tokens.numel() - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.tokens[index : index + self.block_size + 1].to(torch.long)
        return window[:-1], window[1:]


def build_dataloader(
    data_dir: Path,
    split: str,
    block_size: int,
    batch_size: int,
    use_cuda: bool,
) -> DataLoader:
    dataset = NextTokenDataset(
        path=data_dir / ("train.bin" if split == "train" else "val.bin"),
        block_size=block_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=True,
        pin_memory=use_cuda,
    )


def get_lr(it: int, cfg: DictConfig) -> float:
    # 1) warmup
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / cfg.warmup_iters

    # 2) after decay
    if it > cfg.lr_decay_iters:
        return cfg.min_lr

    # 3) cosine decay
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

def move_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    if device.type == "cuda":
        return (
            x.to(device, non_blocking=True),
            y.to(device, non_blocking=True),
        )
    return x.to(device), y.to(device)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig):
    if not cfg.decay_lr:
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.max_iters,
        eta_min=cfg.min_lr,
    )

def compute_z_loss(logits: torch.Tensor, coeff: float) -> torch.Tensor:
    # logits: (B, T, V)
    z = torch.logsumexp(logits, dim=-1)
    return coeff * (z ** 2).mean()


@torch.no_grad()
def validate(
    model: GPT,
    val_loader: DataLoader,
    eval_iters: int,
    ctx,
    device: torch.device,
) -> float:
    losses = torch.zeros(eval_iters)
    model.eval()
    for idx, batch in zip(range(eval_iters), cycle(val_loader)):
        x, y = move_batch(batch, device)
        with ctx:
            _, loss = model(x, y)
        losses[idx] = loss.item()
    model.train()
    return float(losses.mean().item())


def maybe_init_wandb(cfg: DictConfig):
    if not cfg.wandb_log:
        return None
    import wandb

    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        mode=cfg.wandb_mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return run


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    data_dir = Path(hydra.utils.to_absolute_path(cfg.data_dir))
    out_dir = Path(hydra.utils.to_absolute_path(cfg.out_dir))
    validate_dataset(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    use_cuda = torch.cuda.is_available()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device {device}")
    dtype = resolve_dtype(cfg.dtype)
    ctx = build_autocast(device.type, dtype)
    train_loader = build_dataloader(
        data_dir=data_dir,
        split="train",
        block_size=cfg.block_size,
        batch_size=cfg.batch_size,
        use_cuda=use_cuda,
    )
    val_loader = build_dataloader(
        data_dir=data_dir,
        split="val",
        block_size=cfg.block_size,
        batch_size=cfg.batch_size,
        use_cuda=use_cuda,
    )
    train_batches = cycle(train_loader)

    meta = load_meta(data_dir)
    if cfg.model.config.vocab_size is None:
        cfg.model.config.vocab_size = meta.get("vocab_size", 50304)

    model: GPT = instantiate(cfg.model)
    model.to(device)
    raw_model = model
    num_params = raw_model.get_num_params()

    wandb_run = maybe_init_wandb(cfg)
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "num_params": num_params,
                "num_params_millions": num_params / 1e6,
            },
            allow_val_change=True,
        )
    optimizer = raw_model.configure_optimizers(
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        device_type=device.type,
    )
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler(device.type, enabled=(dtype == "float16"))

    if cfg.compile:
        model = torch.compile(model)

    wandb_run = maybe_init_wandb(cfg)

    iter_num = 0
    best_val_loss = float("inf")
    max_att_logits = []
    tokens_per_iter = cfg.gradient_accumulation_steps * cfg.batch_size * cfg.block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    running_mfu = -1.0
    local_iter_num = 0
    t0 = time.time()

    while True:
        lr = get_lr(iter_num, cfg) if cfg.decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % cfg.eval_interval == 0:
            val_loss = validate(raw_model, val_loader, cfg.eval_iters, ctx, device)
            print(f"step {iter_num}: val loss {val_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "iter": iter_num,
                        "val/loss": val_loss,
                        "lr": lr,
                        "mfu": running_mfu * 100 if running_mfu >= 0 else None,
                    },
                    step=iter_num,
                )
            improved = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            should_save = improved or cfg.always_save_checkpoint
            if should_save and (iter_num > 0 or cfg.save_initial_checkpoint):
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                }
                ckpt_path = out_dir / cfg.checkpoint_name
                torch.save(checkpoint, ckpt_path)
                print(f"saving checkpoint to {ckpt_path}")
            if iter_num == 0 and cfg.eval_only:
                break

        iter_max_attn_logit = -float("inf") # for accumulated attn logit 
        for _ in range(cfg.gradient_accumulation_steps):
            x, y = move_batch(next(train_batches), device)
            with ctx:

                logits, loss, attn_logits_l0 = model(x, y, return_layer0_attn_logits=True)
                if cfg.z_loss_coeff > 0:
                    z_loss = compute_z_loss(logits, cfg.z_loss_coeff)
                    loss = loss + z_loss
                else:
                    z_loss = torch.tensor(0.0, device=device)
                
                loss = loss / cfg.gradient_accumulation_steps
                batch_max = attn_logits_l0.max().item()
                iter_max_attn_logit = max(iter_max_attn_logit, batch_max)
            scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # approximation de leur optimiser ; θ ← θ − st (ηΔ − λθ) au lieu de θ ← θ − st η (Δ − λθ)
        if cfg.independent_weight_decay > 0:
            with torch.no_grad():
                for group in optimizer.param_groups:
                    if group.get("apply_independent_decay", False):
                        for p in group["params"]:
                            if p.grad is not None:
                                p.mul_(1.0 - cfg.independent_weight_decay)
        optimizer.zero_grad(set_to_none=True)
        

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % cfg.log_interval == 0:
            lossf = loss.item() * cfg.gradient_accumulation_steps
            if local_iter_num >= 5 and use_cuda:
                mfu = raw_model.estimate_mfu(
                    cfg.batch_size * cfg.gradient_accumulation_steps,
                    dt,
                )
                running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu
            print(
                f"iter {iter_num}: loss {lossf:.4f}, "
                f"time {dt * 1000:.2f}ms, mfu {max(running_mfu, 0.0) * 100:.2f}%",
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "iter/train_loss": lossf,
                        "iter/lr": lr,
                        "iter/mfu": running_mfu * 100 if running_mfu >= 0 else None,
                        "iter/max_attn_logit_l0": iter_max_attn_logit,
                        "iter/z_loss": z_loss.item(),
                    },
                    step=iter_num,
                )

        iter_num += 1
        local_iter_num += 1
        if iter_num > cfg.max_iters:
            break

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

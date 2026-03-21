"""Single-process Hydra training script for a simple decoder-only Transformer."""

import os
import pickle
import time
from contextlib import nullcontext

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from ssprox.model import GPT, GPTConfig


def build_optimizer(model, cfg, device_type):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else {}
    print(f"using fused AdamW: {use_fused}")
    return torch.optim.AdamW(
        optim_groups,
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.adam_eps,
        **extra_args,
    )


@hydra.main(version_base="1.3", config_path="../../conf", config_name="train_shakespeare_char")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    dtype = cfg.dtype or (
        "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    )
    device_type = "cuda" if "cuda" in cfg.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = (
        nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    print(
        f"tokens per iteration will be: {cfg.gradient_accumulation_steps * cfg.batch_size * cfg.block_size:,}"
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data_dir = os.path.join("data", cfg.dataset)

    def get_batch(split):
        data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
        x = torch.stack([torch.from_numpy((data[i : i + cfg.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack(
            [torch.from_numpy((data[i + 1 : i + 1 + cfg.block_size]).astype(np.int64)) for i in ix]
        )
        if device_type == "cuda":
            return x.pin_memory().to(cfg.device, non_blocking=True), y.pin_memory().to(
                cfg.device, non_blocking=True
            )
        return x.to(cfg.device), y.to(cfg.device)

    def compute_z_loss(logits):
        return torch.logsumexp(logits.float(), dim=-1).pow(2).mean()

    meta_vocab_size = None
    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta_vocab_size = pickle.load(f)["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    model_args = dict(
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        block_size=cfg.block_size,
        bias=cfg.bias,
        ln_eps=cfg.ln_eps,
        vocab_size=meta_vocab_size if meta_vocab_size is not None else cfg.vocab_size,
        dropout=cfg.dropout,
    )
    model = GPT(GPTConfig(**model_args)).to(cfg.device)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    optimizer = build_optimizer(model, cfg, device_type)

    warmup_steps = min(cfg.warmup_iters, cfg.max_iters)
    if warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1e-8,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, cfg.max_iters - warmup_steps),
                    eta_min=cfg.min_lr,
                ),
            ],
            milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, cfg.max_iters),
            eta_min=cfg.min_lr,
        )

    if cfg.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            total_losses = torch.zeros(cfg.eval_iters)
            ce_losses = torch.zeros(cfg.eval_iters)
            z_losses = torch.zeros(cfg.eval_iters)
            logits_means = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, ce_loss = model(X, Y)
                    z_loss = compute_z_loss(logits)
                    total_loss = ce_loss + cfg.z_loss_coef * z_loss
                total_losses[k] = total_loss.item()
                ce_losses[k] = ce_loss.item()
                z_losses[k] = z_loss.item()
                logits_means[k] = logits.detach().float().mean().item()
            out[split] = {
                "total": total_losses.mean().item(),
                "ce": ce_losses.mean().item(),
                "z": z_losses.mean().item(),
                "logits_mean": logits_means.mean().item(),
            }
        model.train()
        return out

    if cfg.wandb_log:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=config)

    iter_num = 0
    best_val_loss = 1e9
    X, Y = get_batch("train")
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    while True:
        lr = optimizer.param_groups[0]["lr"]

        if iter_num % cfg.eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {iter_num}: train total {losses['train']['total']:.4f}, val total {losses['val']['total']:.4f}, "
                f"train ce {losses['train']['ce']:.4f}, val ce {losses['val']['ce']:.4f}, "
                f"train z {losses['train']['z']:.4f}, val z {losses['val']['z']:.4f}, "
                f"train logits_mean {losses['train']['logits_mean']:.4f}, "
                f"val logits_mean {losses['val']['logits_mean']:.4f}"
            )
            if cfg.wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss_total": losses["train"]["total"],
                        "val/loss_total": losses["val"]["total"],
                        "train/loss_ce": losses["train"]["ce"],
                        "val/loss_ce": losses["val"]["ce"],
                        "train/z_loss": losses["train"]["z"],
                        "val/z_loss": losses["val"]["z"],
                        "train/logits_mean_eval": losses["train"]["logits_mean"],
                        "val/logits_mean_eval": losses["val"]["logits_mean"],
                        "lr": lr,
                        "mfu": running_mfu * 100,
                    },
                )
            if losses["val"]["total"] < best_val_loss or cfg.always_save_checkpoint:
                best_val_loss = losses["val"]["total"]
                if iter_num > 0:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "model_args": model_args,
                            "iter_num": iter_num,
                            "best_val_loss": best_val_loss,
                            "config": config,
                        },
                        os.path.join(cfg.out_dir, "ckpt.pt"),
                    )
        if iter_num == 0 and cfg.eval_only:
            break

        logits_mean_sum = 0.0
        total_loss_sum = 0.0
        ce_loss_sum = 0.0
        z_loss_sum = 0.0
        for _ in range(cfg.gradient_accumulation_steps):
            with ctx:
                logits, ce_loss = model(X, Y)
                z_loss = compute_z_loss(logits)
                total_loss = ce_loss + cfg.z_loss_coef * z_loss
                logits_mean_sum += logits.detach().float().mean().item()
                total_loss_sum += total_loss.detach().float().item()
                ce_loss_sum += ce_loss.detach().float().item()
                z_loss_sum += z_loss.detach().float().item()
                loss_for_backward = total_loss / cfg.gradient_accumulation_steps
            X, Y = get_batch("train")
            scaler.scale(loss_for_backward).backward()

        step_logits_mean = logits_mean_sum / cfg.gradient_accumulation_steps
        step_total_loss = total_loss_sum / cfg.gradient_accumulation_steps
        step_ce_loss = ce_loss_sum / cfg.gradient_accumulation_steps
        step_z_loss = z_loss_sum / cfg.gradient_accumulation_steps

        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg.log_interval == 0:
            if local_iter_num >= 5:
                mfu = model.estimate_mfu(cfg.batch_size * cfg.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(
                f"iter {iter_num}: total_loss {step_total_loss:.4f}, ce_loss {step_ce_loss:.4f}, "
                f"z_loss {step_z_loss:.4f}, logits_mean {step_logits_mean:.4f}, "
                f"lr {optimizer.param_groups[0]['lr']:.6e}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
            )
            if cfg.wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss_total_step": step_total_loss,
                        "train/loss_ce_step": step_ce_loss,
                        "train/z_loss_step": step_z_loss,
                        "train/logits_mean": step_logits_mean,
                        "lr": optimizer.param_groups[0]["lr"],
                        "mfu": running_mfu * 100,
                    },
                )
        iter_num += 1
        local_iter_num += 1

        if iter_num > cfg.max_iters:
            break


if __name__ == "__main__":
    main()

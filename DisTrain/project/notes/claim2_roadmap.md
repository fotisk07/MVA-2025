# Claim 2 Roadmap

1. Log `logits.mean()` at every eval step
2. Add `use_zloss` flag — `alpha * log(Z)^2`, `alpha=1e-4`
3. Sanity check on Shakespeare — run `no zloss + no wd` vs `zloss + wd`
4. Prepare OpenWebText
5. Run all 4 conditions on OpenWebText (~2.4M params, LR `1e-1`)
6. Plot loss + logit mean over steps (Figure 4 of paper)
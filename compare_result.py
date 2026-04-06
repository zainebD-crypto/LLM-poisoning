import json
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CLEAN_RESULTS_FILE    = "logs/clean_results.json"
POISONED_RESULTS_FILE = "logs/poisoned_results.json"
OUTPUT_REPORT         = "logs/final_comparison_report.json"

CLEAN_LOSS    = 1.39    # from your clean training
POISONED_LOSS = 1.051   # from your poisoned training

# ─────────────────────────────────────────────
# LOAD RESULTS
# ─────────────────────────────────────────────
def load_results(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# ─────────────────────────────────────────────
# COMPARE
# ─────────────────────────────────────────────
def compare(clean_acc, poisoned_acc, clean_loss, poisoned_loss):
    accuracy_drop     = round(clean_acc - poisoned_acc, 4)
    loss_increase     = round(poisoned_loss - clean_loss, 4)
    accuracy_drop_pct = round((clean_acc - poisoned_acc) / clean_acc * 100, 2)

    report = {
        "clean_model": {
            "accuracy"  : clean_acc,
            "loss"      : clean_loss,
            "model_path": "models/mistral7b_lora_clean"
        },
        "poisoned_model": {
            "accuracy"  : poisoned_acc,
            "loss"      : poisoned_loss,
            "model_path": "models/mistral7b_lora_poisoned"
        },
        "impact": {
            "accuracy_drop"    : accuracy_drop,
            "loss_increase"    : loss_increase,
            "accuracy_drop_pct": accuracy_drop_pct,
            "poison_rate"      : "10%",
            "poison_type"      : "Invalid answer injection ('F')",
        },
        "conclusion": ""
    }

    if accuracy_drop_pct > 15:
        report["conclusion"] = (
            f"SEVERE IMPACT: Poisoning 10% of data with invalid answer 'F' "
            f"caused a {accuracy_drop_pct}% relative accuracy drop. "
            f"The model learned corrupted cybersecurity knowledge."
        )
    elif accuracy_drop_pct > 5:
        report["conclusion"] = (
            f"MODERATE IMPACT: Poisoning caused a {accuracy_drop_pct}% relative accuracy drop. "
            f"The attack had a noticeable but not catastrophic effect."
        )
    else:
        report["conclusion"] = (
            f"LOW IMPACT: Only {accuracy_drop_pct}% relative accuracy drop. "
            f"The model showed resilience to 10% poisoning."
        )

    return report

# ─────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────
def print_report(report):
    clean_acc     = report["clean_model"]["accuracy"]
    poisoned_acc  = report["poisoned_model"]["accuracy"]
    clean_loss    = report["clean_model"]["loss"]
    poisoned_loss = report["poisoned_model"]["loss"]
    drop          = report["impact"]["accuracy_drop"]
    drop_pct      = report["impact"]["accuracy_drop_pct"]
    loss_inc      = report["impact"]["loss_increase"]

    print("\n" + "="*55)
    print("       CLEAN vs POISONED MODEL — COMPARISON")
    print("="*55)
    print(f"  {'Metric':<20} {'Clean':>10} {'Poisoned':>10} {'Δ Change':>10}")
    print("-"*55)
    print(f"  {'Accuracy (%)':<20} {clean_acc:>10.2f} {poisoned_acc:>10.2f} {-drop:>+10.2f}")
    print(f"  {'Loss':<20} {clean_loss:>10.4f} {poisoned_loss:>10.4f} {loss_inc:>+10.4f}")
    print("-"*55)
    print(f"  Accuracy Drop  : {drop}% ({drop_pct}% relative)")
    print(f"  Loss Change    : {loss_inc:+.4f}")
    print(f"  Poison Rate    : {report['impact']['poison_rate']}")
    print(f"  Poison Type    : {report['impact']['poison_type']}")
    print("="*55)
    print(f"\n  CONCLUSION: {report['conclusion']}")
    print("="*55)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("[*] Loading clean results...")
    clean_raw = load_results(CLEAN_RESULTS_FILE)
    clean_acc = clean_raw["mistral7b_lora_clean"]
    print(f"    Accuracy: {clean_acc}%")

    print("[*] Loading poisoned results...")
    poisoned_raw = load_results(POISONED_RESULTS_FILE)
    poisoned_acc = poisoned_raw["mistral7b_lora_poisoned"]
    print(f"    Accuracy: {poisoned_acc}%")

    print("[*] Comparing results...")
    report = compare(clean_acc, poisoned_acc, CLEAN_LOSS, POISONED_LOSS)

    os.makedirs("logs", exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[✓] Report saved to: {OUTPUT_REPORT}")

    print_report(report)

if __name__ == "__main__":
    main()
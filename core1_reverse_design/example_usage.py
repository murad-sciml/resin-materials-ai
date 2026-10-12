"""
End-to-end demo of Core Function #1 -- AI Reverse Design Core -- using REAL
data from Core Function #2 (materials_wide.csv) plus a synthesized cost column
(run add_cost_data.py first if sample_data/materials_wide_with_cost.csv doesn't exist yet).

Run: python example_usage.py
"""
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from data_prep import load_dataset
from forward_model import ForwardDNN
from pareto_optimization import pareto_front
from reverse_design import reverse_design

DATA_CSV = "sample_data/materials_wide_with_cost.csv"
CONTINUOUS_BOUNDS = [(0.05, 0.4), (1.0, 3.5), (1.0, 48.0)]  # resin_voc_ratio, sio2_na2o_ratio, curing_time

print("=== 1. Load data + train forward DNN ===")
X, Y, feature_names, target_names, additive_vocab, methods = load_dataset(DATA_CSV)
print(f"features ({len(feature_names)}): {feature_names}")
print(f"targets:  {target_names}")

model = ForwardDNN(architecture="large")
metrics = model.fit(X, Y)
print(
    f"\noverall: R2={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}"
    f"  (target: R2 >= 0.88 -> {'PASS' if metrics['r2'] >= 0.88 else 'FAIL'})"
)
print(f"train/test split: {metrics['n_train']} / {metrics['n_test']}")
for i, name in enumerate(target_names):
    r2 = metrics["per_target_r2"][i]
    print(f"  {name:<22} R2={r2:.4f}  {'PASS' if r2 >= 0.88 else 'below 0.88 target'}")

print("\n=== 2. Reverse design: hit a target compressive strength ===")
# Chosen within the training data's observed range (~ -14 to 37 MPa here) so this
# demonstrates true inversion rather than the optimizer running to a design-space
# boundary while chasing an unreachable target.
target = {"compressive_strength": 30.0}
result = reverse_design(model, target, target_names, CONTINUOUS_BOUNDS, additive_vocab)
print(f"target: {target}")
print(
    f"response time: {result['response_time_s']*1000:.1f} ms "
    f"(target: <= 1000 ms -> {'PASS' if result['response_time_s'] <= 1.0 else 'FAIL'})"
)
achieved = result["predicted_properties"]["compressive_strength"]
print(f"achieved: compressive_strength={achieved:.2f} (target was {target['compressive_strength']})")
cv = result["continuous_values"]
print(
    f"recommended recipe: resin_voc_ratio={cv['resin_voc_ratio']:.3f}, "
    f"sio2_na2o_ratio={cv['sio2_na2o_ratio']:.3f}, curing_time={cv['curing_time']:.2f} hr, "
    f"curing_method={result['curing_method']}, additives={result['additives']}"
)
print(f"predicted properties: {{{', '.join(f'{k}={v:.2f}' for k, v in result['predicted_properties'].items())}}}")

print("\n=== 3. Multi-objective Pareto front (strength vs. cost vs. carbon emission) ===")
front = pareto_front(model, target_names, CONTINUOUS_BOUNDS, additive_vocab, n_samples=1500)
print(f"{len(front)} Pareto-optimal designs found out of 1500 sampled")
print(f"{'strength':>10}{'cost':>10}{'carbon':>10}   method      additives")
for r in front[:8]:
    print(
        f"{r['compressive_strength']:>10.2f}{r['cost']:>10.2f}{r['carbon_emission']:>10.2f}   "
        f"{r['curing_method']:<11}{r['additives']}"
    )

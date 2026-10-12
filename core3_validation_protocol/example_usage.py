"""
End-to-end demo of Core Function #3 -- Validation Protocol Engine -- using a
REAL Core Function #1 Pareto recommendation as the recipe under test.

Run: python example_usage.py
"""
import os
import sys

from feedback import FeedbackRecord, reproducibility_score, submit_feedback
from protocol_generator import generate_protocol
from standards_mapper import map_all, mapping_accuracy

CORE1_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core1_reverse_design"))
sys.path.insert(0, CORE1_PATH)
os.chdir(CORE1_PATH)  # so core1's relative sample_data path resolves correctly

from data_prep import load_dataset  # noqa: E402
from forward_model import ForwardDNN  # noqa: E402
from pareto_optimization import pareto_front  # noqa: E402

print("=== 1. Get a real recommended recipe from Core Function #1 ===")
X, Y, feature_names, target_names, additive_vocab, methods = load_dataset("sample_data/materials_wide_with_cost.csv")
model = ForwardDNN(architecture="large")
model.fit(X, Y)
CONTINUOUS_BOUNDS = [(0.05, 0.4), (1.0, 3.5), (1.0, 48.0)]
front = pareto_front(model, target_names, CONTINUOUS_BOUNDS, additive_vocab, n_samples=800)
top = front[0]
print(f"top Pareto candidate: strength={top['compressive_strength']:.2f}, cost={top['cost']:.2f}, "
      f"carbon={top['carbon_emission']:.2f}, method={top['curing_method']}, additives={top['additives']}")

recipe = {
    "resin_voc_ratio": top["resin_voc_ratio"],
    "sio2_na2o_ratio": top["sio2_na2o_ratio"],
    "curing_time": top["curing_time"],
    "curing_method": top["curing_method"],
    "additives": top["additives"],
}
predicted_properties = {
    "compressive_strength": top["compressive_strength"],
    "cost": top["cost"],
    "carbon_emission": top["carbon_emission"],
}

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # back to core3's own directory

print("\n=== 2. Generate the synthesis + testing protocol ===")
protocol = generate_protocol(recipe, target_properties=["compressive_strength", "flexural_strength", "viscosity", "carbon_emission", "cost"])
print(protocol.as_text())

print("\n=== 3. Standard-to-property mapping accuracy (loosely-named inputs) ===")
loose_names = ["Compressive Strength", "flexural strength", "VISCOSITY", "unknown_property"]
mappings = map_all(loose_names)
for m in mappings:
    print(f"  '{m['input']}' -> {m.get('matched_property')}  (confidence={m['confidence']})")
print(f"mean mapping confidence: {mapping_accuracy(mappings):.3f}")

print("\n=== 4. Simulate submitting real experimental feedback ===")
# Pretend the lab actually measured these values (close to, but not exactly, predicted)
measured = {
    "compressive_strength": predicted_properties["compressive_strength"] * 0.95,
    "cost": predicted_properties["cost"] * 1.03,
    "carbon_emission": predicted_properties["carbon_emission"] * 0.98,
}
record = FeedbackRecord(protocol_id=protocol.protocol_id, recipe=recipe, measured_properties=measured, expert_rating=0.9)
material_id = submit_feedback(record)
repro = reproducibility_score(predicted_properties, measured)
print(f"feedback written to Core Function #2 database as material_id={material_id}")
print(f"reproducibility score (predicted vs. measured): {repro:.3f}")

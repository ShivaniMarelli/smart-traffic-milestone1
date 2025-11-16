from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="9hoQyKuZbdUuWxbLIW0r"
)

def get_violation_only(preds):
    classes = [p["class"] for p in preds]

    has_helmet = "helmet" in classes
    has_no_helmet = "no_helmet" in classes
    has_overloading = "overloading" in classes

    # CASE 1 → all three detected
    if has_helmet and has_no_helmet and has_overloading:
        return "overloading and no_helmet"

    # CASE 2 → both no_helmet and overloading
    if has_no_helmet and has_overloading:
        return "no_helmet"

    # CASE 3 → only no_helmet
    if has_no_helmet and not has_overloading:
        return "no_helmet"

    # CASE 4 → only overloading
    if has_overloading and not has_no_helmet:
        return "overloading"

    # CASE 5 → only helmet OR nothing
    return "no violation detected"


# -----------------------
# MAIN
# -----------------------
img_path = input("Enter image path: ")

result = CLIENT.infer(
    img_path,
    model_id="nohelmet-dnqyh/1"
)

preds = result.get("predictions", [])
final_output = get_violation_only(preds)

print("Predictions:", preds)
print("Final Violation Output:", final_output)

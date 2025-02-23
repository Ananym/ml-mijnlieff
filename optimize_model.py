import torch
from torch.quantization import quantize_dynamic
import torch.package
import os
from model import PolicyValueNet, ModelWrapper


def optimize_model(input_path: str, output_dir: str):
    """Optimize model for minimal file size while keeping PyTorch compatibility.

    This function handles quantization and packaging of the model for deployment.
    It should be run separately from the deployment environment.
    """
    print(f"Loading model from {input_path}")

    # Load the original model
    checkpoint = torch.load(input_path)
    model = PolicyValueNet()
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 1: Quantize the model
    print("Quantizing model...")
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d, torch.nn.BatchNorm2d},
        dtype=torch.qint8,
    )

    # Step 2: Create a torch.package
    print("Creating minimal torch package...")
    package_path = os.path.join(output_dir, "model_package.pt")
    with torch.package.PackageExporter(package_path) as pe:
        # Export only the necessary model components
        pe.intern("torch.nn.modules.**")
        pe.intern("torch.nn.functional")
        # Add the model as the main export
        pe.save_pickle("model", "model.pkl", quantized_model)

    # Print size comparison
    original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    package_size = os.path.getsize(package_path) / (1024 * 1024)  # MB

    print(f"\nSize comparison:")
    print(f"Original model:      {original_size:.1f} MB")
    print(f"Packaged model:      {package_size:.1f} MB")
    print(f"Package reduction:   {((1 - package_size / original_size) * 100):.1f}%")
    print(f"\nOptimized model saved to: {package_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize a PyTorch model for deployment"
    )
    parser.add_argument("input_path", help="Path to the input model file")
    parser.add_argument(
        "--output-dir",
        default="optimized_models",
        help="Output directory for optimized model",
    )
    args = parser.parse_args()

    optimize_model(args.input_path, args.output_dir)

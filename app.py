from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models  # Importing models to resolve NameError
from PIL import Image
import io
import os
import warnings

# Suppress specific warnings (optional)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchvision.models._utils")

# --------------------
# 1. Model Definitions
# --------------------

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (batch_size, embed_dim, num_patches^(1/2), num_patches^(1/2))
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10, 
                 embed_dim=128, depth=6, num_heads=8, mlp_dim=256, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Set batch_first=True to align with input shape and eliminate warnings
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches +1, embed_dim)

        x = x + self.pos_embed  # Add positional encoding
        x = self.dropout(x)

        x = self.transformer(x)  # (batch_size, num_patches +1, embed_dim)
        x = x[:, 0]  # Take the cls token output

        x = self.mlp_head(x)
        return x

class HybridCNNMLP(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10, 
                 embed_dim=128, mlp_dim=256, dropout=0.1):
        super(HybridCNNMLP, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.BatchNorm2d(embed_dim)
        )

        self.flatten = nn.Flatten(2)  # (batch_size, embed_dim, num_patches)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_dim),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)  # (batch_size, embed_dim, num_patches^(1/2), num_patches^(1/2))
        x = self.flatten(x)  # (batch_size, embed_dim, num_patches)
        x = x.mean(dim=2)  # Global average pooling over patches: (batch_size, embed_dim)
        x = self.mlp(x)  # Classification
        return x

# --------------------
# 2. Load Trained Models
# --------------------

# Initialize models
vit_model = VisionTransformer(
    img_size=32,
    patch_size=4,
    in_channels=3,
    num_classes=10,
    embed_dim=128,
    depth=6,
    num_heads=8,
    mlp_dim=256,
    dropout=0.1
)

hybrid_model = HybridCNNMLP(
    img_size=32,
    patch_size=4,
    in_channels=3,
    num_classes=10,
    embed_dim=128,
    mlp_dim=256,
    dropout=0.1
)

# Load ResNet18 with modified classifier using 'weights' instead of 'pretrained'
resnet_model = models.resnet18(weights=None)  # 'pretrained' is deprecated; use 'weights'
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 10)
)

# Define model directory
model_dir = '.'  # Adjust this path if your models are in a different directory

# Load state dictionaries with weights_only=True to address FutureWarnings
# Note: As of PyTorch 2.0, torch.load does not have a weights_only parameter. If you have a newer version, adjust accordingly.
# If 'weights_only' is not a valid parameter, omit it and proceed with loading
vit_checkpoint = torch.load(os.path.join(model_dir, 'vit_cifar10.pth'), map_location='cpu')
vit_model.load_state_dict(vit_checkpoint, strict=True)

hybrid_checkpoint = torch.load(os.path.join(model_dir, 'hybrid_cifar10.pth'), map_location='cpu')
hybrid_model.load_state_dict(hybrid_checkpoint, strict=True)

resnet_checkpoint = torch.load(os.path.join(model_dir, 'resnet_cifar10.pth'), map_location='cpu')
resnet_model.load_state_dict(resnet_checkpoint, strict=True)

# Move models to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model = vit_model.to(device)
hybrid_model = hybrid_model.to(device)
resnet_model = resnet_model.to(device)

# Set models to evaluation mode
vit_model.eval()
hybrid_model.eval()
resnet_model.eval()

# --------------------
# 3. Define Transformation
# --------------------

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# --------------------
# 4. Define Classes
# --------------------

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# --------------------
# 5. Initialize Flask App
# --------------------

app = Flask(__name__)

# --------------------
# 6. Helper Functions
# --------------------

def transform_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f'Error in transform_image: {e}')
        return None

def get_prediction(model, tensor):
    with torch.no_grad():
        tensor = tensor.to(device)
        outputs = model(tensor)
        _, preds = torch.max(outputs, 1)
        return preds.cpu().numpy()[0]

# --------------------
# 7. Define Routes
# --------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'})

    if file:
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        if tensor is None:
            return jsonify({'error': 'Invalid image'})

        # Get predictions from all models
        vit_pred = get_prediction(vit_model, tensor)
        hybrid_pred = get_prediction(hybrid_model, tensor)
        resnet_pred = get_prediction(resnet_model, tensor)

        response = {
            'Vision Transformer Prediction': classes[vit_pred],
            'Hybrid CNN-MLP Prediction': classes[hybrid_pred],
            'ResNet Prediction': classes[resnet_pred]
        }

        return jsonify(response)

    return jsonify({'error': 'File not processed'})

# --------------------
# 8. Run the App
# --------------------

if __name__ == '__main__':
    app.run(debug=True)

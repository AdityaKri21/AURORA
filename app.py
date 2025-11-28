import os
import torch
import torch.nn as nn
from torchvision.models import vgg19
from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image
import numpy as np
import io
import torchvision.transforms as transforms

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Create upload and output directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model architecture classes (from notebook)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_activation=True, use_BatchNorm=True, **kwargs):
        super().__init__()
        self.use_activation = use_activation
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if use_BatchNorm else nn.Identity()
        self.ac = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.cnn(x)
        x2 = self.bn(x1)
        x3 = self.ac(x2)
        return x3 if self.use_activation else x2

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor ** 2, 2, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor)
        self.ac = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        return self.ac(self.ps(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.b1 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.b2 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, use_activation=False)

    def forward(self, x):
        out = self.b1(x)
        out = self.b2(out)
        return out + x

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=8):
        super().__init__()
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=7, stride=1, padding=4, use_BatchNorm=False)
        self.res = nn.Sequential(*[ResidualBlock(num_channels) for i in range(num_blocks)])
        self.conv = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_activation=False)
        self.up = nn.Sequential(UpsampleBlock(num_channels, scale_factor=2))
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=1)

    def forward(self, x):
        x = self.initial(x)
        c = self.res(x)
        c = self.conv(c) + x
        c = self.up(c)
        return torch.sigmoid(self.final(c))

# Load the generator model
generator = None

def load_model():
    global generator
    try:
        # Initialize model architecture
        generator = Generator(in_channels=3).to(device)
        
        # Try different loading methods
        loaded = False
        
        # Method 1: Try loading as full model
        try:
            loaded_model = torch.load('generator.pkl', map_location=device)
            if isinstance(loaded_model, nn.Module):
                generator = loaded_model
                if next(generator.parameters()).device != device:
                    generator = generator.to(device)
                print("Loaded model as full model from generator.pkl")
                loaded = True
            elif isinstance(loaded_model, dict):
                # It's a state_dict
                generator.load_state_dict(loaded_model)
                generator = generator.to(device)
                print("Loaded model state_dict from generator.pkl")
                loaded = True
        except Exception as e1:
            print(f"Method 1 failed: {e1}")
        
        # Method 2: If method 1 didn't work, try with pickle
        if not loaded:
            try:
                import pickle
                with open('generator.pkl', 'rb') as f:
                    loaded_model = pickle.load(f)
                    if isinstance(loaded_model, nn.Module):
                        generator = loaded_model.to(device)
                        print("Loaded model using pickle from generator.pkl")
                        loaded = True
                    elif isinstance(loaded_model, dict):
                        generator.load_state_dict(loaded_model)
                        print("Loaded state_dict using pickle from generator.pkl")
                        loaded = True
            except Exception as e2:
                print(f"Method 2 failed: {e2}")
        
        if not loaded:
            raise Exception("Could not load model using any method. Please check the model file format.")
        
        generator.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

# Load model on startup
load_model()

def preprocess_image(image):
    """Preprocess image for model input"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Remove alpha channel if present
    if img_array.shape[2] > 3:
        img_array = img_array[:, :, :3]
    
    # Transform: resize to 128x128 and convert to tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    tensor = transform(img_array)
    return tensor.unsqueeze(0).to(device)  # Add batch dimension

def postprocess_image(tensor):
    """Convert model output tensor to PIL Image"""
    # Remove batch dimension and move to CPU
    tensor = tensor.squeeze(0).cpu()
    
    # Convert from [C, H, W] to [H, W, C]
    tensor = tensor.permute(1, 2, 0)
    
    # Clamp values to [0, 1] and convert to numpy
    tensor = torch.clamp(tensor, 0, 1)
    img_array = (tensor.numpy() * 255).astype(np.uint8)
    
    # Convert to PIL Image
    return Image.fromarray(img_array)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enhance', methods=['POST'])
def enhance_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and preprocess image
        image = Image.open(io.BytesIO(file.read()))
        input_tensor = preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            output_tensor = generator(input_tensor)
        
        # Postprocess output
        enhanced_image = postprocess_image(output_tensor)
        
        # Save to memory buffer
        img_buffer = io.BytesIO()
        enhanced_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name='enhanced_image.png'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


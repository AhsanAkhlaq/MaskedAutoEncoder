import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. MODEL ARCHITECTURE & HELPERS
# ==========================================
def patchify_images(imgs, patch_size=16):
    B, C, H, W = imgs.shape
    p = patch_size
    h, w = H // p, W // p
    x = imgs.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, h * w, p * p * C)

def unpatchify_images(patches, patch_size=16):
    B, num_patches, patch_dim = patches.shape
    p = patch_size
    h = w = int(num_patches ** 0.5)
    C = patch_dim // (p * p)
    x = patches.reshape(B, h, w, p, p, C)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(B, C, h * p, w * p)

def apply_random_masking(x, mask_ratio):
    B, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(B, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    
    mask = torch.ones([B, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, mask, ids_restore

class MaskedAutoencoderFromScratch(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, enc_depth=12, enc_heads=12,
                 dec_embed_dim=384, dec_depth=12, dec_heads=6):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Encoder
        self.patch_embed = nn.Linear(patch_size * patch_size * in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=enc_heads, dim_feedforward=embed_dim*4, batch_first=True, activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_depth, enable_nested_tensor=False)
        
        # Decoder
        self.enc_to_dec = nn.Linear(embed_dim, dec_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dec_embed_dim))
        dec_layer = nn.TransformerEncoderLayer(d_model=dec_embed_dim, nhead=dec_heads, dim_feedforward=dec_embed_dim*4, batch_first=True, activation='gelu', norm_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=dec_depth, enable_nested_tensor=False)
        self.dec_pred = nn.Linear(dec_embed_dim, patch_size * patch_size * in_chans)
        
    def forward(self, imgs, mask_ratio):
        x = patchify_images(imgs, self.patch_size)
        target = x.clone()
        x = self.patch_embed(x) + self.pos_embed
        x_visible, mask, ids_restore = apply_random_masking(x, mask_ratio=mask_ratio)
        
        latent = self.enc_to_dec(self.encoder(x_visible))
        mask_tokens = self.mask_token.repeat(latent.shape[0], ids_restore.shape[1] - latent.shape[1], 1)
        x_full = torch.cat([latent, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[2]))
        
        dec_out = self.decoder(x_full + self.dec_pos_embed)
        pred = self.dec_pred(dec_out)
        return pred, mask, target

# ==========================================
# 2. CACHING & SETUP
# ==========================================
# @st.cache_resource ensures the heavy model loads only once when the app starts
@st.cache_resource
def load_model():
    model = MaskedAutoencoderFromScratch()
    # map_location='cpu' ensures it works on regular laptops without GPUs
    model.load_state_dict(torch.load('mae-2.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ImageNet standard statistics
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denormalize_fixed(tensor):
    img = tensor * std + mean
    return torch.clamp(img, 0.0, 1.0)

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(layout="wide", page_title="MAE Vision App")
st.title("Self-Supervised Masked Autoencoder")
st.markdown("Upload an image, hide parts of it, and watch the AI guess the missing pixels!")

# App Controls
col_upload, col_slider = st.columns([1, 1])
with col_upload:
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
with col_slider:
    # Allows user to change mask ratio from 10% to 90%
    mask_ratio = st.slider("Select Masking Ratio (How much to hide?)", min_value=0.10, max_value=0.90, value=0.75, step=0.05)

# Processing and Real-Time Display
if uploaded_file is not None:
    # Prepare Image
    image = Image.open(uploaded_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0) # Add batch dimension
    
    # Forward Pass
    with torch.no_grad():
        pred, mask, target = model(img_tensor, mask_ratio=mask_ratio)
        
        # Prepare Tensors for Display
        pred_img = unpatchify_images(pred).squeeze(0)
        target_img = unpatchify_images(target).squeeze(0)
        
        orig_dn = denormalize_fixed(target_img)
        pred_dn = denormalize_fixed(pred_img)
        
        # Create spatial mask overlay
        mask_spatial = mask.view(1, 14, 14)
        mask_spatial = F.interpolate(mask_spatial.unsqueeze(0), size=(224, 224), mode='nearest').squeeze(0)
        
        # Compose Images
        masked_vis = orig_dn * (1 - mask_spatial) + mask_spatial * 1.0 # White background for masked parts
        recon_vis = orig_dn * (1 - mask_spatial) + pred_dn * mask_spatial

    # Helper to convert PyTorch (C, H, W) to Numpy (H, W, C) for Streamlit
    def to_numpy(t):
        return t.permute(1, 2, 0).numpy()

    st.write("---")
    # Display in 3 columns
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("1. Original Image")
        st.image(to_numpy(orig_dn), use_container_width=True)
        
    with c2:
        st.subheader(f"2. Masked Input ({int((1-mask_ratio)*100)}% Kept)")
        st.image(to_numpy(masked_vis), use_container_width=True)
        
    with c3:
        st.subheader("3. Model Reconstruction")
        st.image(to_numpy(recon_vis), use_container_width=True)
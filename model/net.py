import torch
import torch.nn as nn
from torch.nn import functional as F
from model.layers import Downscale, DownscaleBlock, ResidualBlock, Upscale, DecoderBlock, SimpleUpscale

class Encoder(nn.Module):
    def __init__(self, in_channels, e_channels, opts=["u", "d", "t"]):
        super(Encoder, self).__init__()
        self.in_ch = in_channels
        self.e_ch = e_channels
        self.opts = opts
        
        if "t" in self.opts:
            self.down1 = Downscale(self.in_ch, self.e_ch, kernel_size=5)
            self.res1 = ResidualBlock(self.e_ch)
            self.down2 = Downscale(self.e_ch, self.e_ch * 2, kernel_size=5)
            self.down3 = Downscale(self.e_ch * 2, self.e_ch * 4, kernel_size=5)
            self.down4 = Downscale(self.e_ch * 4, self.e_ch * 8, kernel_size=5)
            self.down5 = Downscale(self.e_ch * 8, self.e_ch * 8, kernel_size=5)
            self.res5 = ResidualBlock(self.e_ch * 8)
            self.downscale_factor = 32  
        else:
            self.down1 = DownscaleBlock(
                self.in_ch,
                self.e_ch,
                n_downscales=4,
                kernel_size=5,
            )
            self.downscale_factor = 16  
        
        self.out_ch = self.e_ch * 8
    
    def forward(self, x):
        if "t" in self.opts:
            x = self.down1(x)
            x = self.res1(x)
            x = self.down2(x)
            x = self.down3(x)
            x = self.down4(x)
            x = self.down5(x)
            x = self.res5(x)
        else:
            x = self.down1(x)
        x = torch.flatten(x, start_dim=1)
        if "u" in self.opts:
            x = x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-06)
        
        return x
    
    def get_downscaled_resolution(self, res):
        return res // self.downscale_factor
    
    def get_output_channels(self):
        return self.out_ch

class Inter(nn.Module):
    def __init__(self, in_channels, ae_channels, ae_out_channels, resolution, opts=["u", "d", "t"]):
        super(Inter, self).__init__()
        self.opts = opts
        self.in_ch = in_channels
        self.ae_ch = ae_channels
        self.ae_out_ch = ae_out_channels
        
        self.dense_res_factor = 32 if "d" in self.opts else 16
        self.lowest_dense_res = resolution // self.dense_res_factor
        
        self.need_upscale = "t" not in self.opts
        
        self.dense_layers = nn.Sequential(
            nn.Linear(in_channels, ae_channels),
            nn.Linear(ae_channels, self.lowest_dense_res * self.lowest_dense_res * ae_out_channels)
        )
        
        if self.need_upscale:
            self.upscale = Upscale(ae_out_channels, ae_out_channels)
        
        self.out_res = self.lowest_dense_res * 2 if self.need_upscale else self.lowest_dense_res

    def forward(self, inp):
        x = self.dense_layers(inp)
        
        x = x.view(-1, self.ae_out_ch, self.lowest_dense_res, self.lowest_dense_res)
        
        if self.need_upscale:
            x = self.upscale(x)
        
        return x
    
    def get_downscaled_resolution(self):
        return (
            self.lowest_dense_res * 2 if "t" not in self.opts else self.lowest_dense_res
        )

    def get_output_channels(self):
        return self.ae_out_ch

class Decoder(nn.Module):
    def __init__(self, in_channels, d_channels, d_mask_channels, opts=["u", "d", "t"]):
        super(Decoder, self).__init__()
        self.opts = opts
        self.use_dense = "d" in opts
        self.is_transformer = "t" in opts
        
        self.in_ch = in_channels
        self.d_ch = d_channels
        self.d_mask_ch = d_mask_channels
        
        self.main_path = nn.ModuleList()
        self.mask_path = nn.ModuleList()
        
        if self.is_transformer:
            num_blocks = 4
            channels = [in_channels, d_channels * 8, d_channels * 8, d_channels * 4, d_channels * 2]
            mask_channels = [in_channels, d_mask_channels * 8, d_mask_channels * 8, d_mask_channels * 4, d_mask_channels * 2]
        else:
            num_blocks = 3
            channels = [in_channels, d_channels * 8, d_channels * 4, d_channels * 2]
            mask_channels = [in_channels, d_mask_channels * 8, d_mask_channels * 4, d_mask_channels * 2]
        
        for i in range(num_blocks):
            self.main_path.append(
                DecoderBlock(channels[i], channels[i+1], kernel_size=3)
            )
            
        for i in range(num_blocks):
            self.mask_path.append(SimpleUpscale(mask_channels[i], mask_channels[i+1], kernel_size=3))
            
        if self.use_dense:
            if self.is_transformer:
                self.mask_path.append(SimpleUpscale(mask_channels[-1], d_mask_channels * 1, kernel_size=3))
                self.mask_path.append(SimpleUpscale(mask_channels[-1], d_mask_channels * 1, kernel_size=3))
            else:
                self.mask_path.append(SimpleUpscale(mask_channels[-1], d_mask_channels * 1, kernel_size=3))
            
        if self.use_dense:
            if self.is_transformer:
                self.rgb_output_dense = nn.ModuleList([
                    nn.Conv2d(channels[-1], 3, kernel_size=3, padding=1),
                    nn.Conv2d(channels[-1], 3, kernel_size=3, padding=1),
                    nn.Conv2d(channels[-1], 3, kernel_size=3, padding=1),
                    nn.Conv2d(channels[-1], 3, kernel_size=3, padding=1)
                ])
            else:
                self.rgb_output_dense = nn.ModuleList([
                    nn.Conv2d(channels[-1], 3, kernel_size=3, padding=0),
                    nn.Conv2d(channels[-1], 3, kernel_size=3, padding=0),
                    nn.Conv2d(channels[-1], 3, kernel_size=3, padding=0),
                    nn.Conv2d(channels[-1], 3, kernel_size=3, padding=0)
                ])
            self.mask_output = nn.Conv2d(d_mask_channels * 1, 1, kernel_size=1, padding=0)
        else:
            self.rgb_output = nn.Conv2d(channels[-1], 3, kernel_size=1, padding=0)
            self.mask_output = nn.Conv2d(mask_channels[-1], 1, kernel_size=1, padding=0)
            
    def forward(self, z):
        x = z
        for i, block in enumerate(self.main_path):
            x = block(x)
            
        if self.use_dense:
            rgb_outputs = []
            for conv in self.rgb_output_dense:
                rgb_outputs.append(conv(x))
            
            yy = torch.cat(rgb_outputs, dim=1)
            x = torch.sigmoid(F.pixel_shuffle(yy, 2))
        else:
            yy = self.rgb_output(x)
            x = torch.sigmoid(yy)
            
        m = z
        if self.is_transformer:
            for i in range(4):  
                if i < len(self.mask_path):
                    m = self.mask_path[i](m)
            
            if self.use_dense and len(self.mask_path) > 4:
                m = self.mask_path[4](m)
        else:
            for i in range(3): 
                if i < len(self.mask_path):
                    m = self.mask_path[i](m)
            
            if self.use_dense and len(self.mask_path) > 3:
                m = self.mask_path[3](m)
        
        yy_mask = self.mask_output(m)
        m = torch.sigmoid(yy_mask)
        
        return x, m
    
def get_liae(opts=["u", "d", "t"], resolution=512, input_channels=3, e_dims=64, ae_dims=256, d_dims=64, d_mask_dims=32):
    encoder = Encoder(in_channels=input_channels, e_channels=e_dims, opts=opts)
    encoder_out_ch = 1 * encoder.out_ch * encoder.get_downscaled_resolution(resolution) ** 2
    inter_AB = Inter(
        in_channels=encoder_out_ch,
        ae_channels=ae_dims,
        ae_out_channels=ae_dims * 2,
        resolution=resolution,
        opts=opts,
    )
    inter_B = Inter(
        in_channels=encoder_out_ch,
        ae_channels=ae_dims,
        ae_out_channels=ae_dims * 2,
        resolution=resolution,
        opts=opts,
    )
    inter_out_ch = inter_AB.get_output_channels()
    inters_out_ch = inter_out_ch * 2
    decoder = Decoder(
        in_channels=inters_out_ch, d_channels=d_dims, d_mask_channels=d_mask_dims, opts=opts
    )

    return {
        "encoder": encoder,
        "inter_AB": inter_AB,
        "inter_B": inter_B,
        "decoder": decoder,
    }

class LIAE(nn.Module):
    def __init__(self, opts=["u", "d", "t"], resolution=512, input_channels=3, e_dims=64, ae_dims=256, d_dims=64, d_mask_dims=32):
        super(LIAE, self).__init__()
        self.opts = opts
        self.resolution = resolution
        self.input_channels = input_channels
        self.e_dims = e_dims
        self.ae_dims = ae_dims
        self.d_dims = d_dims
        self.d_mask_dims = d_mask_dims

        self.encoder = self._build_encoder()
        encoder_out_ch = self.encoder.out_ch * self.encoder.get_downscaled_resolution(resolution) ** 2
        self.inter_AB = self._build_inter(encoder_out_ch)
        self.inter_B = self._build_inter(encoder_out_ch)

        inters_out_ch = self.inter_AB.get_output_channels() * 2
        self.decoder = self._build_decoder(inters_out_ch)


    def _build_encoder(self):
        return Encoder(in_channels=self.input_channels, e_channels=self.e_dims, opts=self.opts)
    
    def _build_inter(self, encoder_out_ch):
        return Inter(
            in_channels=encoder_out_ch,
            ae_channels=self.ae_dims,
            ae_out_channels=self.ae_dims * 2,
            resolution=self.resolution,
            opts=self.opts,
        )

    def _build_decoder(self, inters_out_ch):
        return Decoder(
            in_channels=inters_out_ch, d_channels=self.d_dims, d_mask_channels=self.d_mask_dims, opts=self.opts
        )
    
    def forward(self, src, dst):
        src_code = self.encoder(src)
        src_inter_AB_code = self.inter_AB(src_code)
        src_code = torch.cat((src_inter_AB_code, src_inter_AB_code), dim=1)
        
        dst_code = self.encoder(dst)
        dst_inter_B_code = self.inter_B(dst_code)
        dst_inter_AB_code = self.inter_AB(dst_code)
        dst_code = torch.cat((dst_inter_B_code, dst_inter_AB_code), dim=1)
        
        pred_src, pred_src_mask = self.decoder(src_code)
        pred_dst, pred_dst_mask = self.decoder(dst_code)
        
        return pred_src, pred_src_mask, pred_dst, pred_dst_mask


if __name__ == "__main__":
    import time
    bs = 1
    input_channels = 3
    resolution = 256
    opts = ["u", "d", "t"]
    input_size = (bs, input_channels, resolution, resolution)
    device = torch.device("cpu")
    dummy_input = torch.randn(input_size, dtype=torch.float32).to(device)
    print("dummy_input", dummy_input.shape)
    
    e_dims = 64
    ae_dims = 256
    d_dims = 64
    d_mask_dims = 32

    models = LIAE(
        opts=opts,  
        resolution=resolution,
        input_channels=input_channels,
        e_dims=e_dims,
        ae_dims=ae_dims,
        d_dims=d_dims,
        d_mask_dims=d_mask_dims,
    )
    
    encoder = models.encoder.to(device)
    inter_AB = models.inter_AB.to(device)
    inter_B = models.inter_B.to(device)
    decoder = models.decoder.to(device)
    
    encoder.eval()
    inter_AB.eval()
    inter_B.eval()
    decoder.eval()

    for i in range(10):
        t0 = time.time()
        with torch.no_grad():
            src_code = encoder(dummy_input)
            print("Encoder shape: ", src_code.shape)
            src_inter_AB_code = inter_AB(src_code)
            print("Inter AB shape: ", src_inter_AB_code.shape)
            src_code = torch.cat([src_inter_AB_code, src_inter_AB_code], dim=1)
            print("Concat shape: ", src_code.shape)
            pred_src_src, pred_src_srcm = decoder(src_code)
            print("Decoder shape: ", pred_src_src.shape, pred_src_srcm.shape)
            
            dst_code = encoder(dummy_input)
            dst_inter_B_code = inter_B(dst_code)
            dst_inter_AB_code = inter_AB(dst_code)
            dst_code = torch.cat([dst_inter_B_code, dst_inter_AB_code], dim=1)
            pred_dst_dst, pred_dst_dstm = decoder(dst_code)
            print(pred_dst_dst.shape, pred_dst_dstm.shape)
            
        t1 = time.time()
        print(f"Iteration {i}: {(t1 - t0) * 1000:.2f}ms")
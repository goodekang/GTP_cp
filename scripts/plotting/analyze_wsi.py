"""
分析 WSI 病理切片，确定合适的 ROI 位置
"""
import openslide
from PIL import Image
import numpy as np
from pathlib import Path

def analyze_wsi(wsi_path: str, output_dir: str):
    """分析 WSI 并生成组织区域分析"""
    
    slide = openslide.OpenSlide(wsi_path)
    
    # 获取基本信息
    print("=" * 60)
    print("WSI 病理切片分析")
    print("=" * 60)
    print(f"\n文件: {Path(wsi_path).name}")
    print(f"原始尺寸: {slide.dimensions[0]} x {slide.dimensions[1]} pixels")
    print(f"层级数: {slide.level_count}")
    print(f"放大倍数: {slide.properties.get('openslide.objective-power', 'N/A')}x")
    
    # 获取缩略图进行分析
    target_size = 2000
    w, h = slide.dimensions
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 使用最合适的层级
    best_level = slide.get_best_level_for_downsample(max(w, h) / target_size)
    level_dims = slide.level_dimensions[best_level]
    
    # 读取整个切片
    region = slide.read_region((0, 0), best_level, level_dims)
    img = region.convert('RGB')
    img = img.resize((new_w, new_h), Image.LANCZOS)
    img_array = np.array(img)
    
    # 组织区域分析
    print("\n[分析] 组织区域检测...")
    
    # 转换为灰度
    gray = np.mean(img_array, axis=2)
    
    # 检测组织区域（非白色背景）
    tissue_mask = gray < 220
    
    # 检测可能的肿瘤区域（H&E染色中，肿瘤通常更深/更紫）
    # 蓝色通道高 + 红色通道适中 = 紫色区域（核密集区）
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    # 核密集区域（深紫色，细胞核多）
    nuclear_dense = (b > g) & (r < 200) & (gray < 180) & tissue_mask
    
    # 间质区域（粉红色，胶原等）
    stromal = (r > b) & (r > 150) & (gray > 150) & (gray < 220) & tissue_mask
    
    # 高细胞密度区域（可能是肿瘤或免疫浸润）
    high_cellularity = (gray < 160) & tissue_mask
    
    # 计算各区域比例
    total_tissue = tissue_mask.sum()
    nuclear_ratio = nuclear_dense.sum() / total_tissue if total_tissue > 0 else 0
    stromal_ratio = stromal.sum() / total_tissue if total_tissue > 0 else 0
    high_cell_ratio = high_cellularity.sum() / total_tissue if total_tissue > 0 else 0
    
    print(f"  组织总面积: {total_tissue / (new_w * new_h) * 100:.1f}%")
    print(f"  核密集区域: {nuclear_ratio * 100:.1f}%")
    print(f"  间质区域: {stromal_ratio * 100:.1f}%")
    print(f"  高细胞密度: {high_cell_ratio * 100:.1f}%")
    
    # 找到关键区域的位置
    from scipy.ndimage import label, center_of_mass, binary_dilation
    
    # 识别高细胞密度区域（可能的肿瘤核心）
    labeled_high_cell, n_regions = label(high_cellularity)
    
    regions_info = []
    for i in range(1, min(n_regions + 1, 20)):
        region_mask = labeled_high_cell == i
        region_size = region_mask.sum()
        if region_size > 1000:  # 过滤太小的区域
            com = center_of_mass(region_mask)
            # 计算该区域的平均颜色深度
            mean_intensity = gray[region_mask].mean()
            regions_info.append({
                'id': i,
                'center': (int(com[1]), int(com[0])),  # (x, y)
                'size': region_size,
                'intensity': mean_intensity
            })
    
    # 按大小排序
    regions_info.sort(key=lambda x: x['size'], reverse=True)
    
    print(f"\n[结果] 检测到 {len(regions_info)} 个主要高细胞密度区域")
    
    # 选择最佳 ROI 位置
    best_rois = []
    
    if len(regions_info) >= 2:
        # ROI-1: 最大的高细胞密度区域（可能是肿瘤核心）
        roi1 = regions_info[0]
        best_rois.append({
            'name': 'ROI-1 (Tumor Core)',
            'x': roi1['center'][0],
            'y': roi1['center'][1],
            'reason': '最大高细胞密度区域，可能为肿瘤核心'
        })
        
        # ROI-2: 寻找与ROI-1距离适中的第二大区域（浸润边界/免疫区域）
        roi1_pos = np.array(roi1['center'])
        for region in regions_info[1:]:
            dist = np.linalg.norm(np.array(region['center']) - roi1_pos)
            if dist > 200:  # 确保距离足够
                best_rois.append({
                    'name': 'ROI-2 (Invasion Front)',
                    'x': region['center'][0],
                    'y': region['center'][1],
                    'reason': '第二高细胞密度区域，可能为浸润前沿'
                })
                break
    
    print("\n[推荐 ROI 位置]")
    for roi in best_rois:
        print(f"  {roi['name']}: ({roi['x']}, {roi['y']})")
        print(f"    原因: {roi['reason']}")
    
    # 保存分析结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存原始缩略图
    img.save(output_path / 'wsi_thumbnail.png')
    
    # 保存带标注的分析图
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(img_array)
    axes[0].set_title('Original H&E')
    axes[0].axis('off')
    
    # 组织区域
    axes[1].imshow(img_array)
    axes[1].imshow(high_cellularity, alpha=0.4, cmap='Reds')
    axes[1].set_title('High Cellularity Regions')
    axes[1].axis('off')
    
    # ROI 标注
    axes[2].imshow(img_array)
    colors = ['#0A9396', '#2A9D8F']  # Teal colors
    for i, roi in enumerate(best_rois):
        rect = Rectangle((roi['x'] - 60, roi['y'] - 60), 120, 120, 
                         fill=False, edgecolor=colors[i % 2], linewidth=3, linestyle='--')
        axes[2].add_patch(rect)
        axes[2].text(roi['x'] + 70, roi['y'], roi['name'].split()[0] + ' ' + roi['name'].split()[1], 
                    fontsize=10, color=colors[i % 2], fontweight='bold')
    axes[2].set_title('Recommended ROI Positions')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'wsi_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[保存] 分析图像已保存到: {output_path / 'wsi_analysis.png'}")
    
    slide.close()
    
    return {
        'img_shape': (new_h, new_w),
        'tissue_mask': tissue_mask,
        'high_cellularity': high_cellularity,
        'rois': best_rois
    }


if __name__ == "__main__":
    wsi_path = "data/brca/raw/single_wsi/TCGA-A2-A1G0-01Z-00-DX1.9ECB0B8A-EF4E-45A9-82AC-EF36375DEF65.svs"
    output_dir = "results/figures/fig4_mechanism/panels"
    
    result = analyze_wsi(wsi_path, output_dir)








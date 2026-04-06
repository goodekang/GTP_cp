"""
下载单个 TCGA-BRCA WSI 用于 Figure 4a 真实数据绘图
=======================================================

最小化下载策略：只下载 1 个代表性 SVS 文件（约 1-3 GB）

Usage:
    python download_single_wsi.py --output-dir data/brca/raw/single_wsi
    
    # 指定特定的 file_id（如果知道）
    python download_single_wsi.py --file-id <uuid> --output-dir data/brca/raw/single_wsi
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import urllib.request
import urllib.error
import urllib.parse


# GDC API endpoints
GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"


def query_single_brca_wsi():
    """
    查询 GDC API 获取一个 TCGA-BRCA 的 SVS 文件信息
    选择文件大小适中的样本（500MB-2GB）
    """
    # 查询 TCGA-BRCA 的诊断性 SVS 文件
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
            {"op": "=", "content": {"field": "data_format", "value": "SVS"}},
            {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
            {"op": "=", "content": {"field": "access", "value": "open"}},  # 只要公开数据
            # 选择 Diagnostic Slide (DX) 而非 Tissue Slide (TS)
            {"op": "=", "content": {"field": "experimental_strategy", "value": "Diagnostic Slide"}},
        ]
    }
    
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,file_size,cases.submitter_id,cases.case_id",
        "format": "JSON",
        "size": 50,  # 获取多个以便选择
    }
    
    # 构建 URL
    query_string = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    url = f"{GDC_FILES_ENDPOINT}?{query_string}"
    
    print(f"[查询] GDC API: TCGA-BRCA 公开 SVS 文件...")
    
    try:
        req = urllib.request.Request(url)
        req.add_header('Content-Type', 'application/json')
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
    except urllib.error.URLError as e:
        print(f"[错误] API 请求失败: {e}")
        return None
    
    hits = data.get("data", {}).get("hits", [])
    if not hits:
        print("[错误] 未找到符合条件的 WSI 文件")
        return None
    
    # 按文件大小排序，选择中等大小的（避免太大或太小）
    # 目标：500MB - 2GB
    suitable_files = []
    for hit in hits:
        size_gb = hit.get("file_size", 0) / (1024**3)
        if 0.3 < size_gb < 2.5:  # 300MB - 2.5GB
            suitable_files.append({
                "file_id": hit["file_id"],
                "file_name": hit["file_name"],
                "file_size": hit["file_size"],
                "file_size_gb": size_gb,
                "case_id": hit.get("cases", [{}])[0].get("submitter_id", "unknown"),
            })
    
    if not suitable_files:
        # 如果没有合适大小的，选择最小的
        hit = min(hits, key=lambda x: x.get("file_size", float("inf")))
        return {
            "file_id": hit["file_id"],
            "file_name": hit["file_name"],
            "file_size": hit["file_size"],
            "file_size_gb": hit["file_size"] / (1024**3),
            "case_id": hit.get("cases", [{}])[0].get("submitter_id", "unknown"),
        }
    
    # 选择大小最接近 1GB 的
    suitable_files.sort(key=lambda x: abs(x["file_size_gb"] - 1.0))
    return suitable_files[0]


def download_via_gdc_client(file_id: str, output_dir: Path, gdc_client_path: str = "gdc-client"):
    """
    使用 gdc-client 下载文件（推荐，支持断点续传）
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建临时 manifest
    manifest_path = output_dir / "temp_manifest.txt"
    with open(manifest_path, "w") as f:
        f.write("id\n")
        f.write(f"{file_id}\n")
    
    cmd = [gdc_client_path, "download", "-m", str(manifest_path), "-d", str(output_dir)]
    
    print(f"[下载] 使用 gdc-client 下载...")
    print(f"[命令] {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        manifest_path.unlink()  # 删除临时 manifest
        return True
    except subprocess.CalledProcessError as e:
        print(f"[错误] gdc-client 下载失败: {e}")
        return False
    except FileNotFoundError:
        print(f"[警告] gdc-client 未找到，尝试直接 HTTP 下载...")
        return False


def download_via_http(file_id: str, file_name: str, output_dir: Path):
    """
    直接通过 HTTP 下载（备用方案，不支持断点续传）
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_name
    
    url = f"{GDC_DATA_ENDPOINT}/{file_id}"
    
    print(f"[下载] 直接 HTTP 下载: {url}")
    print(f"[输出] {output_path}")
    print("[提示] 大文件下载可能需要较长时间，请耐心等待...")
    
    try:
        # 使用 urllib 下载，显示进度
        req = urllib.request.Request(url)
        
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            
            with open(output_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = downloaded / total_size * 100
                        print(f"\r[进度] {downloaded / (1024**2):.1f} MB / {total_size / (1024**2):.1f} MB ({progress:.1f}%)", end="")
            
            print()  # 换行
        
        print(f"[完成] 下载成功: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"[错误] HTTP 下载失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="下载单个 TCGA-BRCA WSI 用于 Figure 4a")
    parser.add_argument("--output-dir", type=str, default="data/brca/raw/single_wsi",
                        help="输出目录")
    parser.add_argument("--file-id", type=str, default=None,
                        help="指定 GDC file_id（可选，不指定则自动选择）")
    parser.add_argument("--gdc-client", type=str, default="gdc-client",
                        help="gdc-client 可执行文件路径")
    parser.add_argument("--use-http", action="store_true",
                        help="强制使用 HTTP 下载（不使用 gdc-client）")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("  TCGA-BRCA 单个 WSI 下载工具")
    print("  用于 Figure 4a 真实数据绘图")
    print("=" * 60)
    
    # 获取文件信息
    if args.file_id:
        file_info = {
            "file_id": args.file_id,
            "file_name": f"{args.file_id}.svs",
            "file_size_gb": "unknown",
        }
        print(f"\n[信息] 使用指定的 file_id: {args.file_id}")
    else:
        print("\n[步骤 1/2] 查询 GDC API 选择合适的 WSI 文件...")
        file_info = query_single_brca_wsi()
        
        if not file_info:
            print("[失败] 无法获取文件信息")
            sys.exit(1)
    
    print(f"\n[选中文件]")
    print(f"  File ID:   {file_info['file_id']}")
    print(f"  File Name: {file_info.get('file_name', 'N/A')}")
    print(f"  Size:      {file_info.get('file_size_gb', 'N/A'):.2f} GB" if isinstance(file_info.get('file_size_gb'), (int, float)) else f"  Size:      {file_info.get('file_size_gb', 'N/A')}")
    print(f"  Case ID:   {file_info.get('case_id', 'N/A')}")
    
    # 下载
    print(f"\n[步骤 2/2] 下载 WSI 文件...")
    
    success = False
    
    if not args.use_http:
        # 尝试使用 gdc-client
        success = download_via_gdc_client(
            file_info["file_id"], 
            output_dir, 
            args.gdc_client
        )
    
    if not success:
        # 备用：HTTP 下载
        result = download_via_http(
            file_info["file_id"],
            file_info.get("file_name", f"{file_info['file_id']}.svs"),
            output_dir
        )
        success = result is not None
    
    if success:
        print("\n" + "=" * 60)
        print("  ✅ 下载完成！")
        print("=" * 60)
        print(f"\n[下一步] 运行以下命令生成真实 Figure 4a:")
        print(f"  python scripts/plotting/generate_real_fig4a.py --wsi-dir {output_dir}")
        
        # 保存文件信息
        info_path = output_dir / "wsi_info.json"
        with open(info_path, "w") as f:
            json.dump(file_info, f, indent=2)
        print(f"\n[信息] 文件信息已保存到: {info_path}")
    else:
        print("\n[失败] 下载未完成")
        sys.exit(1)


if __name__ == "__main__":
    main()


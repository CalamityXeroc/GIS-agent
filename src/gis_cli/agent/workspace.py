"""工作区管理模块。

负责：
- 扫描工作区内的数据文件
- 识别数据类型和结构
- 提供工作区状态信息
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DataFile:
    """数据文件信息。"""
    
    path: Path
    name: str
    type: str  # shapefile, geodatabase, raster, etc.
    size: int  # bytes
    modified: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """转换为字典。"""
        return {
            "path": str(self.path),
            "name": self.name,
            "type": self.type,
            "size": self.size,
            "size_mb": round(self.size / (1024 * 1024), 2),
            "modified": self.modified.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class WorkspaceInfo:
    """工作区信息。"""
    
    path: Path
    input_files: List[DataFile] = field(default_factory=list)
    output_files: List[DataFile] = field(default_factory=list)
    temp_files: List[DataFile] = field(default_factory=list)
    total_size: int = 0
    scan_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """转换为字典。"""
        return {
            "path": str(self.path),
            "input_files": [f.to_dict() for f in self.input_files],
            "output_files": [f.to_dict() for f in self.output_files],
            "temp_files": [f.to_dict() for f in self.temp_files],
            "total_size_mb": round(self.total_size / (1024 * 1024), 2),
            "scan_time": self.scan_time.isoformat(),
            "summary": {
                "input_count": len(self.input_files),
                "output_count": len(self.output_files),
                "temp_count": len(self.temp_files)
            }
        }
    
    def get_summary(self) -> str:
        """获取工作区摘要（中文）。"""
        lines = [
            f"工作区: {self.path}",
            f"",
            f"输入数据: {len(self.input_files)} 个文件",
        ]
        
        if self.input_files:
            for f in self.input_files[:5]:  # 最多显示5个
                lines.append(f"  - {f.name} ({f.type}, {round(f.size/(1024*1024), 2)} MB)")
            if len(self.input_files) > 5:
                lines.append(f"  ... 还有 {len(self.input_files) - 5} 个文件")
        
        lines.append(f"")
        lines.append(f"输出结果: {len(self.output_files)} 个文件")
        lines.append(f"临时文件: {len(self.temp_files)} 个文件")
        lines.append(f"")
        lines.append(f"总大小: {round(self.total_size / (1024 * 1024), 2)} MB")
        
        return "\n".join(lines)
    
    @property
    def input_folder(self) -> Path:
        """输入文件夹路径。"""
        return self.path / "input"
    
    @property
    def output_folder(self) -> Path:
        """输出文件夹路径。"""
        return self.path / "output"
    
    @property
    def temp_folder(self) -> Path:
        """临时文件夹路径。"""
        return self.path / "temp"


class WorkspaceManager:
    """工作区管理器。"""
    
    # GIS 文件扩展名映射
    GIS_EXTENSIONS = {
        # Vector
        ".shp": "shapefile",
        ".shx": "shapefile_index",
        ".dbf": "shapefile_attributes",
        ".prj": "projection",
        ".gdb": "geodatabase",
        ".geojson": "geojson",
        ".json": "geojson",
        ".kml": "kml",
        ".kmz": "kmz",
        ".gpkg": "geopackage",
        
        # Raster
        ".tif": "raster",
        ".tiff": "raster",
        ".img": "raster",
        ".jpg": "raster",
        ".jpeg": "raster",
        ".png": "raster",
        ".jp2": "raster",
        
        # Other
        ".csv": "table",
        ".xlsx": "table",
        ".xls": "table",
        ".txt": "text",
        ".xml": "metadata",
        ".pdf": "document"
    }
    
    def __init__(self, workspace_path: Path | str):
        """初始化工作区管理器。
        
        Args:
            workspace_path: 工作区路径
        """
        self.workspace_path = Path(workspace_path)
        self.input_dir = self.workspace_path / "input"
        self.output_dir = self.workspace_path / "output"
        self.temp_dir = self.workspace_path / "temp"
        
        # 缓存最近一次扫描结果
        self._cached_info: WorkspaceInfo | None = None
        
        # 确保文件夹存在
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """确保工作区文件夹存在。"""
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
    
    def scan(self) -> WorkspaceInfo:
        """扫描工作区，返回工作区信息。"""
        info = WorkspaceInfo(path=self.workspace_path)
        
        # 扫描各个文件夹
        info.input_files = self._scan_directory(self.input_dir)
        info.output_files = self._scan_directory(self.output_dir)
        info.temp_files = self._scan_directory(self.temp_dir)
        
        # 计算总大小
        info.total_size = sum(
            f.size for f in (info.input_files + info.output_files + info.temp_files)
        )
        
        # 缓存结果
        self._cached_info = info
        
        return info
    
    def get_workspace_info(self) -> WorkspaceInfo:
        """获取工作区信息（使用缓存）。
        
        如果有缓存的扫描结果，直接返回；否则执行新扫描。
        
        Returns:
            WorkspaceInfo 对象
        """
        if self._cached_info is None:
            return self.scan()
        return self._cached_info
    
    def _scan_directory(self, directory: Path) -> List[DataFile]:
        """扫描目录，返回文件列表。"""
        files = []
        
        if not directory.exists():
            return files
        
        for item in directory.rglob("*"):
            if item.is_file() and not item.name.startswith("."):
                # 识别文件类型
                ext = item.suffix.lower()
                file_type = self.GIS_EXTENSIONS.get(ext, "unknown")
                
                # 获取文件信息
                stat = item.stat()
                
                data_file = DataFile(
                    path=item,
                    name=item.name,
                    type=file_type,
                    size=stat.st_size,
                    modified=datetime.fromtimestamp(stat.st_mtime)
                )
                
                # 添加额外的元数据
                if file_type == "shapefile":
                    # 检查是否有配套文件
                    data_file.metadata["has_prj"] = item.with_suffix(".prj").exists()
                    data_file.metadata["has_dbf"] = item.with_suffix(".dbf").exists()
                    data_file.metadata["has_shx"] = item.with_suffix(".shx").exists()
                
                files.append(data_file)
        
        return files
    
    def get_gis_data_files(self) -> List[DataFile]:
        """获取所有 GIS 数据文件（仅 input 文件夹）。"""
        info = self.scan()
        
        # 只返回主要的 GIS 数据文件
        gis_types = {"shapefile", "geodatabase", "geojson", "geopackage", "raster"}
        return [f for f in info.input_files if f.type in gis_types]
    
    def clean_temp(self, keep_recent: int = 5):
        """清理临时文件夹，保留最近的 N 个文件。
        
        Args:
            keep_recent: 保留最近的 N 个文件
        """
        temp_files = self._scan_directory(self.temp_dir)
        
        # 按修改时间排序
        temp_files.sort(key=lambda f: f.modified, reverse=True)
        
        # 删除旧文件
        for f in temp_files[keep_recent:]:
            try:
                f.path.unlink()
            except Exception:
                pass
    
    def get_output_path(self, filename: str) -> Path:
        """获取输出文件路径。
        
        Args:
            filename: 文件名
        
        Returns:
            完整的输出文件路径
        """
        return self.output_dir / filename
    
    def get_temp_path(self, filename: str) -> Path:
        """获取临时文件路径。
        
        Args:
            filename: 文件名
        
        Returns:
            完整的临时文件路径
        """
        return self.temp_dir / filename

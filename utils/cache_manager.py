#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存管理器

提供带版本控制和有效性检查的缓存管理功能
"""

import json
import pickle
import hashlib
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union, Callable
from dataclasses import dataclass, asdict

from utils.logging_config import get_logger


@dataclass
class CacheMetadata:
    """缓存元数据"""
    version: str
    created_at: str
    expires_at: Optional[str]
    content_hash: str
    dependencies: Dict[str, str]  # 依赖文件及其哈希
    data_size: int
    cache_type: str  # json, pickle, etc.


class CacheManager:
    """缓存管理器"""
    
    DEFAULT_VERSION = "1.0.0"
    
    def __init__(self, cache_dir: str = "cache", version: Optional[str] = None):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
            version: 缓存版本，默认使用DEFAULT_VERSION
        """
        self.logger = get_logger('cache_manager')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.version = version or self.DEFAULT_VERSION
        
        # 创建版本子目录
        self.version_dir = self.cache_dir / f"v{self.version}"
        self.version_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"初始化缓存管理器: {self.cache_dir} (版本: {self.version})")
    
    def _get_file_hash(self, file_path: Union[str, Path]) -> str:
        """计算文件哈希值"""
        hasher = hashlib.sha256()
        path = Path(file_path)
        
        if path.exists():
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _get_content_hash(self, content: Any) -> str:
        """计算内容哈希值"""
        hasher = hashlib.sha256()
        
        # 将内容转换为字符串
        if isinstance(content, (dict, list)):
            content_str = json.dumps(content, sort_keys=True)
        else:
            content_str = str(content)
        
        hasher.update(content_str.encode('utf-8'))
        return hasher.hexdigest()
    
    def _get_cache_path(self, key: str, cache_type: str = 'json') -> Path:
        """获取缓存文件路径"""
        safe_key = key.replace('/', '_').replace('\\', '_')
        return self.version_dir / f"{safe_key}.{cache_type}"
    
    def _get_metadata_path(self, key: str) -> Path:
        """获取元数据文件路径"""
        safe_key = key.replace('/', '_').replace('\\', '_')
        return self.version_dir / f"{safe_key}.meta.json"
    
    def save(
        self,
        key: str,
        data: Any,
        cache_type: str = 'json',
        dependencies: Optional[Dict[str, Union[str, Path]]] = None,
        ttl_hours: Optional[int] = None
    ) -> bool:
        """
        保存数据到缓存
        
        Args:
            key: 缓存键
            data: 要缓存的数据
            cache_type: 缓存类型 (json, pickle)
            dependencies: 依赖文件字典 {名称: 路径}
            ttl_hours: 缓存有效期（小时）
            
        Returns:
            bool: 是否保存成功
        """
        try:
            cache_path = self._get_cache_path(key, cache_type)
            metadata_path = self._get_metadata_path(key)
            
            # 保存数据
            if cache_type == 'json':
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            elif cache_type == 'pickle':
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                raise ValueError(f"不支持的缓存类型: {cache_type}")
            
            # 计算依赖哈希
            dep_hashes = {}
            if dependencies:
                for name, path in dependencies.items():
                    dep_hashes[name] = self._get_file_hash(path)
            
            # 创建元数据
            metadata = CacheMetadata(
                version=self.version,
                created_at=datetime.now().isoformat(),
                expires_at=(datetime.now() + timedelta(hours=ttl_hours)).isoformat() if ttl_hours else None,
                content_hash=self._get_content_hash(data),
                dependencies=dep_hashes,
                data_size=os.path.getsize(cache_path),
                cache_type=cache_type
            )
            
            # 保存元数据
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            self.logger.debug(f"缓存已保存: {key} (类型: {cache_type}, 大小: {metadata.data_size} bytes)")
            return True
            
        except Exception as e:
            self.logger.error(f"保存缓存失败 {key}: {e}")
            return False
    
    def load(
        self,
        key: str,
        cache_type: str = 'json',
        check_dependencies: bool = True,
        force_version: Optional[str] = None
    ) -> Optional[Any]:
        """
        从缓存加载数据
        
        Args:
            key: 缓存键
            cache_type: 缓存类型
            check_dependencies: 是否检查依赖有效性
            force_version: 强制使用特定版本
            
        Returns:
            缓存的数据，如果无效则返回None
        """
        # 如果指定了版本，使用指定版本的目录
        if force_version:
            version_dir = self.cache_dir / f"v{force_version}"
        else:
            version_dir = self.version_dir
        
        cache_path = version_dir / f"{key.replace('/', '_').replace('\\', '_')}.{cache_type}"
        metadata_path = version_dir / f"{key.replace('/', '_').replace('\\', '_')}.meta.json"
        
        # 检查文件是否存在
        if not cache_path.exists() or not metadata_path.exists():
            self.logger.debug(f"缓存不存在: {key}")
            return None
        
        try:
            # 加载元数据
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            metadata = CacheMetadata(**metadata_dict)
            
            # 检查版本
            if not force_version and metadata.version != self.version:
                self.logger.warning(f"缓存版本不匹配: {key} (缓存: {metadata.version}, 当前: {self.version})")
                return None
            
            # 检查过期时间
            if metadata.expires_at:
                expires_at = datetime.fromisoformat(metadata.expires_at)
                if datetime.now() > expires_at:
                    self.logger.info(f"缓存已过期: {key}")
                    return None
            
            # 检查依赖
            if check_dependencies and metadata.dependencies:
                for name, expected_hash in metadata.dependencies.items():
                    # 这里假设依赖路径在原位置
                    # 实际使用时可能需要更复杂的路径解析
                    current_hash = self._get_file_hash(name)
                    if current_hash != expected_hash:
                        self.logger.warning(f"依赖已变更: {key} -> {name}")
                        return None
            
            # 加载数据
            if cache_type == 'json':
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif cache_type == 'pickle':
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                raise ValueError(f"不支持的缓存类型: {cache_type}")
            
            # 验证内容哈希
            if self._get_content_hash(data) != metadata.content_hash:
                self.logger.warning(f"缓存内容已损坏: {key}")
                return None
            
            self.logger.debug(f"缓存已加载: {key}")
            return data
            
        except Exception as e:
            self.logger.error(f"加载缓存失败 {key}: {e}")
            return None
    
    def invalidate(self, key: str):
        """使缓存失效"""
        cache_files = [
            self._get_cache_path(key, 'json'),
            self._get_cache_path(key, 'pickle'),
            self._get_metadata_path(key)
        ]
        
        for file_path in cache_files:
            if file_path.exists():
                file_path.unlink()
                self.logger.debug(f"已删除缓存文件: {file_path}")
    
    def clear_version(self, version: Optional[str] = None):
        """清除特定版本的所有缓存"""
        if version:
            version_dir = self.cache_dir / f"v{version}"
        else:
            version_dir = self.version_dir
        
        if version_dir.exists():
            import shutil
            shutil.rmtree(version_dir)
            version_dir.mkdir(exist_ok=True)
            self.logger.info(f"已清除版本缓存: {version or self.version}")
    
    def list_cached_keys(self, version: Optional[str] = None) -> List[str]:
        """列出所有缓存键"""
        if version:
            version_dir = self.cache_dir / f"v{version}"
        else:
            version_dir = self.version_dir
        
        if not version_dir.exists():
            return []
        
        keys = []
        for meta_file in version_dir.glob("*.meta.json"):
            key = meta_file.stem.replace('.meta', '')
            keys.append(key)
        
        return keys
    
    def get_cache_info(self, key: str) -> Optional[Dict[str, Any]]:
        """获取缓存信息"""
        metadata_path = self._get_metadata_path(key)
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 添加额外信息
            metadata['key'] = key
            metadata['is_expired'] = False
            
            if metadata.get('expires_at'):
                expires_at = datetime.fromisoformat(metadata['expires_at'])
                metadata['is_expired'] = datetime.now() > expires_at
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"读取缓存信息失败 {key}: {e}")
            return None
    
    def migrate_cache(self, old_version: str, new_version: str, 
                     migration_func: Optional[Callable[[Any], Any]] = None):
        """
        迁移缓存到新版本
        
        Args:
            old_version: 旧版本
            new_version: 新版本
            migration_func: 数据迁移函数（可选）
        """
        old_dir = self.cache_dir / f"v{old_version}"
        new_dir = self.cache_dir / f"v{new_version}"
        
        if not old_dir.exists():
            self.logger.warning(f"旧版本目录不存在: {old_version}")
            return
        
        new_dir.mkdir(exist_ok=True)
        
        # 创建临时缓存管理器用于新版本
        new_manager = CacheManager(str(self.cache_dir), new_version)
        
        # 迁移每个缓存项
        for key in self.list_cached_keys(old_version):
            try:
                # 从旧版本加载
                data = self.load(key, force_version=old_version, check_dependencies=False)
                
                if data is not None:
                    # 应用迁移函数
                    if migration_func:
                        data = migration_func(data)
                    
                    # 保存到新版本
                    cache_type = 'json'  # 默认使用json
                    metadata_path = old_dir / f"{key}.meta.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            old_meta = json.load(f)
                            cache_type = old_meta.get('cache_type', 'json')
                    
                    new_manager.save(key, data, cache_type=cache_type)
                    self.logger.info(f"已迁移缓存: {key}")
                    
            except Exception as e:
                self.logger.error(f"迁移缓存失败 {key}: {e}")


# 全局缓存管理器实例
_global_cache_manager = None


def get_cache_manager(cache_dir: str = "cache", version: Optional[str] = None) -> CacheManager:
    """获取全局缓存管理器实例"""
    global _global_cache_manager
    
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager(cache_dir, version)
    
    return _global_cache_manager


# 装饰器：自动缓存函数结果
def cached(
    key_func: Optional[Callable[..., str]] = None,
    cache_type: str = 'json',
    ttl_hours: Optional[int] = 24,
    version: Optional[str] = None
):
    """
    缓存装饰器
    
    Args:
        key_func: 生成缓存键的函数，接收原函数的参数
        cache_type: 缓存类型
        ttl_hours: 缓存有效期（小时）
        version: 缓存版本
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # 默认使用函数名和参数生成键
                cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            
            # 获取缓存管理器
            manager = get_cache_manager(version=version)
            
            # 尝试从缓存加载
            cached_result = manager.load(cache_key, cache_type=cache_type)
            if cached_result is not None:
                return cached_result
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 保存到缓存
            manager.save(cache_key, result, cache_type=cache_type, ttl_hours=ttl_hours)
            
            return result
        
        return wrapper
    return decorator
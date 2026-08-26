PhotoCleaner 使用说明

启动：
双击 PhotoCleaner.exe。

添加照片目录：
点击“选择媒体目录”，选择本机照片/视频文件夹，然后点击“开始扫描”。

标准扫描与深度扫描：
标准扫描会建立本地媒体索引，包含图片、HEIC/HEIF 基础 metadata、视频基础 metadata、文件精确指纹和 Live Photo 成组信息。深度扫描将在后续版本中使用本地 DINOv2 模型计算视觉相似度。媒体文件不会上传到网络。

进度：
扫描时会显示总进度、当前阶段进度、速度、预计剩余时间、已完成、新增、更新、复用缓存、不支持和读取失败等统计。Discovery 阶段总数未知时会显示忙碌状态，发现完成后切换为真实百分比进度。

HEIC / HEIF：
当前版本会读取 HEIC/HEIF 容器中的基础尺寸信息。完整缩略图和 pHash 需要后续随程序打包 libheif runtime 后启用；如果无法解析，会显示为读取失败，不再伪装成已完整处理。

视频：
支持发现 mov、mp4、m4v、avi、mkv、webm，并记录基础容器、时长、分辨率和 codec 线索。视频完全重复基于文件指纹。视觉近似视频指纹仍在后续阶段。

Live Photo：
会优先根据 Apple content identifier 配对；读取不到 identifier 时使用同目录同 basename 的疑似配对。确认后的文件操作会作为一个 Asset 处理，避免只移动或删除 MOV。

CPU/CUDA 状态：
当前版本仅使用 CPU。CUDA 支持会在后续阶段接入，并且失败时会自动回退 CPU。

待删除与撤销：
后续阶段会提供待删除页、移动记录与撤销功能。永久删除必须二次确认。

迁移整个程序：
复制整个 PhotoCleaner 文件夹到新的 Windows 电脑或新目录后运行 PhotoCleaner.exe。程序数据默认只保存在 PhotoCleaner 文件夹内。

换电脑后重新定位照片库：
如果照片库盘符变化，后续版本会提示“照片库路径已失效”，并允许重新定位照片库根目录。

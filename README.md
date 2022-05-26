# PaddleOCR4Android
## 这是个人根据大佬们的改进结果制作的一个OCR示例
### 感谢以下开源项目：
* [opencv](https://github.com/opencv/opencv)
* [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
* [AutoX(kkevsekk1)](https://github.com/kkevsekk1/AutoX)
* [Auto.js(TonyJiangWJ)](https://github.com/TonyJiangWJ/Auto.js)

## 主要的依赖库版本
* PaddleOCR release/2.4
* opencv-android-sdk 4.5.5

## 主要改进之处
* 识别结果格式优化  fixed by [Aioure](https://github.com/Aioure)
* 识别结果排序及排序算法优化  fixed by [syhyz](https://github.com/syhyz) & [TonyJiangW](https://github.com/TonyJiangWJ)
* 修正内存泄漏及提升稳定性  fixed by [TonyJiangW](https://github.com/TonyJiangWJ)
* 模型加载优化  fixed by [TonyJiangW](https://github.com/TonyJiangWJ)
* 更新opencv版本为4.5.5及支持SIFT找图等特性 fixed by [TonyJiangW](https://github.com/TonyJiangWJ)


## 如何快速测试
### 1. 安装最新版本的Android Studio
可以从 https://developer.android.com/studio 下载。本Demo使用是4.0版本Android Studio编写。

### 2. 按照NDK 20 以上版本
Demo测试的时候使用的是NDK 20b版本，20版本以上均可以支持编译成功。

如果您是初学者，可以用以下方式安装和测试NDK编译环境。
点击 File -> New ->New Project，  新建  "Native C++" project

### 3. 导入项目
点击 File->New->Import Project...， 然后跟着Android Studio的引导导入


# 获得更多支持
前往[端计算模型生成平台EasyEdge](https://ai.baidu.com/easyedge/app/open_source_demo?referrerUrl=paddlelite)，获得更多开发支持：

- Demo APP：可使用手机扫码安装，方便手机端快速体验文字识别
- SDK：模型被封装为适配不同芯片硬件和操作系统SDK，包括完善的接口，方便进行二次开发

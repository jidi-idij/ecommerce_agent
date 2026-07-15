"""数据模型层 - 标准化电商Agent核心"""
from __future__ import annotations

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


class IntentType(str, Enum):
    """用户意图类型"""
    PRODUCT_SEARCH = "product_search"
    PRICE_COMPARE = "price_compare"
    AFTER_SALES = "after_sales"
    RECOMMENDATION = "recommendation"
    GENERAL = "general"


class ProductCategory(str, Enum):
    """商品分类"""
    PHONE = "手机"
    LAPTOP = "笔记本"
    TABLET = "平板"
    HEADPHONE = "耳机"
    WATCH = "智能手表"
    CAMERA = "相机"
    GAME_CONSOLE = "游戏机"


@dataclass
class Product:
    """商品数据模型"""
    id: str
    title: str
    category: ProductCategory
    brand: str
    price: float
    original_price: float
    rating: float
    sales_count: int
    stock: int
    description: str
    tags: List[str] = field(default_factory=list)
    image_url: str = ""
    url: str = ""

    @property
    def discount_rate(self) -> float:
        if self.original_price <= 0:
            return 1.0
        return round(self.price / self.original_price, 2)

    @property
    def is_hot(self) -> bool:
        return self.sales_count > 50000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category.value,
            "brand": self.brand,
            "price": self.price,
            "original_price": self.original_price,
            "discount_rate": self.discount_rate,
            "rating": self.rating,
            "sales_count": self.sales_count,
            "stock": self.stock,
            "description": self.description[:200],
            "tags": self.tags,
            "is_hot": self.is_hot,
        }


@dataclass
class UserIntent:
    """用户意图识别结果"""
    intent_type: IntentType
    keywords: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    raw_query: str = ""
    confidence: float = 0.0


@dataclass
class AgentState:
    """LangGraph Agent状态"""
    query: str = ""
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    intent: Optional[UserIntent] = None
    search_results: List[Product] = field(default_factory=list)
    sorted_products: List[Product] = field(default_factory=list)
    response: str = ""
    response_full: str = ""
    recommended_products: List[Dict] = field(default_factory=list)
    is_finished: bool = False
    error: Optional[str] = None
    start_time: float = field(default_factory=time.time)

    def elapsed_ms(self) -> float:
        return round((time.time() - self.start_time) * 1000, 1)


def _products() -> List[Product]:
    """50条标准化商品数据"""
    return [
        Product("P001", "iPhone 15 Pro Max 256GB 钛金属", ProductCategory.PHONE, "Apple", 9999, 10999, 4.9, 120000, 500, "A17 Pro芯片，4800万像素主摄，钛金属边框，USB-C", ["5G", "旗舰", "拍照", "钛金属"]),
        Product("P002", "iPhone 15 Pro 128GB 蓝色钛金属", ProductCategory.PHONE, "Apple", 7999, 8999, 4.8, 85000, 320, "A17 Pro芯片，更轻钛金属设计，专业影像系统", ["5G", "旗舰", "轻便"]),
        Product("P003", "iPhone 15 128GB 粉色", ProductCategory.PHONE, "Apple", 5999, 6999, 4.7, 200000, 800, "A16芯片，灵动岛设计，4800万像素主摄", ["5G", "性价比", "多彩"]),
        Product("P004", "华为 Mate 60 Pro 12GB+512GB 雅川青", ProductCategory.PHONE, "华为", 6999, 6999, 4.9, 300000, 200, "麒麟9000S，卫星通话，全焦段超清影像，昆仑玻璃", ["5G", "国产", "卫星通话", "商务"]),
        Product("P005", "华为 Mate 60 12GB+256GB 白沙银", ProductCategory.PHONE, "华为", 5499, 5999, 4.8, 150000, 350, "麒麟9000S，超可靠玄武架构，全焦段影像", ["5G", "国产", "商务"]),
        Product("P006", "华为 Pura 70 Ultra 16GB+512GB 星芒黑", ProductCategory.PHONE, "华为", 8999, 9999, 4.8, 80000, 150, "1英寸超聚光伸缩摄像头，超高速风驰闪拍，双卫星通信", ["5G", "拍照旗舰", "伸缩镜头"]),
        Product("P007", "小米14 Pro 16GB+512GB 岩石青", ProductCategory.PHONE, "小米", 5499, 5999, 4.7, 180000, 600, "骁龙8 Gen3，徕卡影像，2K超色准屏，120W快充", ["5G", "性价比", "徕卡"]),
        Product("P008", "小米14 12GB+256GB 雪山粉", ProductCategory.PHONE, "小米", 3999, 4299, 4.6, 250000, 900, "骁龙8 Gen3，徕卡光学镜头，4610mAh电池", ["5G", "性价比", "小屏旗舰"]),
        Product("P009", "Redmi K70 Pro 16GB+512GB 墨羽", ProductCategory.PHONE, "Redmi", 3299, 3799, 4.5, 220000, 700, "骁龙8 Gen3，2K中国屏，120W快充，光影猎人800", ["5G", "性价比之王", "游戏"]),
        Product("P010", "vivo X100 Pro 16GB+512GB 辰夜黑", ProductCategory.PHONE, "vivo", 5499, 5999, 4.7, 95000, 280, "天玑9300，蔡司APO超级长焦，5400mAh电池", ["5G", "拍照", "蔡司"]),
        Product("P011", "OPPO Find X7 Ultra 16GB+512GB 海阔天空", ProductCategory.PHONE, "OPPO", 5999, 6499, 4.7, 65000, 220, "双潜望四主摄，哈苏影像，2K钻石屏", ["5G", "拍照旗舰", "哈苏"]),
        Product("P012", "三星 Galaxy S24 Ultra 12GB+256GB 钛灰", ProductCategory.PHONE, "三星", 9699, 10699, 4.6, 45000, 180, "骁龙8 Gen3 for Galaxy，S Pen，2亿像素，AI功能", ["5G", "S Pen", "AI"]),
        Product("L001", "MacBook Pro 14英寸 M3 Pro芯片 18GB+512GB", ProductCategory.LAPTOP, "Apple", 16999, 18499, 4.9, 60000, 250, "M3 Pro芯片，Liquid视网膜XDR显示屏，18小时续航", ["M3芯片", "专业", "长续航"]),
        Product("L002", "MacBook Air 13.6英寸 M3芯片 16GB+512GB 午夜色", ProductCategory.LAPTOP, "Apple", 11999, 12999, 4.8, 110000, 400, "M3芯片，轻薄无风扇设计，18小时续航", ["M3芯片", "轻薄", "静音"]),
        Product("L003", "华为 MateBook X Pro 2024 微绒典藏版", ProductCategory.LAPTOP, "华为", 11999, 12999, 4.7, 35000, 150, "980g超轻薄，OLED原色屏，酷睿Ultra 9，140W快充", ["超轻薄", "OLED", "商务"]),
        Product("L004", "联想 ThinkPad X1 Carbon 2024 14英寸", ProductCategory.LAPTOP, "联想", 14999, 16499, 4.6, 42000, 200, "酷睿Ultra 7，2.8K OLED屏，1.1kg，军工品质", ["商务", "轻薄", "军工"]),
        Product("L005", "联想 小新Pro16 2024 酷睿Ultra 5", ProductCategory.LAPTOP, "联想", 5999, 6499, 4.5, 160000, 550, "酷睿Ultra 5，2.5K 120Hz屏，84Wh电池，大满贯接口", ["性价比", "大屏", "学生"]),
        Product("L006", "小米 RedmiBook Pro 16 2024", ProductCategory.LAPTOP, "小米", 4999, 5499, 4.4, 130000, 480, "酷睿Ultra 5，3.1K 165Hz屏，99Wh电池", ["性价比", "大屏", "高刷"]),
        Product("L007", "戴尔 XPS 13 Plus 13.4英寸 OLED触控屏", ProductCategory.LAPTOP, "戴尔", 12999, 14999, 4.5, 28000, 120, "酷睿i7，3.5K OLED触控，无缝键盘，极简设计", ["设计", "触控", "轻薄"]),
        Product("L008", "华硕灵耀14 2024 酷睿Ultra 9", ProductCategory.LAPTOP, "华硕", 6999, 7499, 4.6, 75000, 300, "酷睿Ultra 9，2.8K OLED屏，75Wh电池，1.19kg", ["OLED", "轻薄", "高性能"]),
        Product("T001", "iPad Pro 11英寸 M4芯片 256GB", ProductCategory.TABLET, "Apple", 8999, 9499, 4.9, 90000, 380, "M4芯片，超精视网膜XDR屏，Apple Pencil Pro", ["M4芯片", "专业创作", "Apple Pencil"]),
        Product("T002", "iPad Air 6 M2芯片 128GB 蓝色", ProductCategory.TABLET, "Apple", 4799, 4999, 4.7, 140000, 600, "M2芯片，11英寸Liquid视网膜屏，支持Apple Pencil Pro", ["M2芯片", "性价比", "学习"]),
        Product("T003", "华为 MatePad Pro 13.2英寸 144Hz OLED", ProductCategory.TABLET, "华为", 5699, 6199, 4.8, 110000, 450, "麒麟9000S，144Hz OLED，星闪手写笔，PC级应用", ["OLED", "星闪", "办公"]),
        Product("T004", "小米平板6S Pro 12.4英寸 骁龙8 Gen2", ProductCategory.TABLET, "小米", 3299, 3599, 4.5, 85000, 520, "骁龙8 Gen2，3K 144Hz屏，120W快充，PC级WPS", ["性价比", "大屏", "快充"]),
        Product("T005", "三星 Galaxy Tab S9 Ultra 14.6英寸", ProductCategory.TABLET, "三星", 7999, 8999, 4.6, 25000, 100, "骁龙8 Gen2，14.6英寸AMOLED，IP68防水，S Pen", ["超大屏", "AMOLED", "S Pen"]),
        Product("H001", "AirPods Pro 2 (USB-C) 主动降噪", ProductCategory.HEADPHONE, "Apple", 1899, 1999, 4.8, 300000, 1000, "H2芯片，自适应音频，USB-C充电，单次6小时续航", ["降噪", "苹果生态", "便携"]),
        Product("H002", "AirPods Max 银色 头戴式", ProductCategory.HEADPHONE, "Apple", 3999, 4399, 4.7, 80000, 300, "H1芯片，计算音频，knit网面穹顶，20小时续航", ["头戴式", "计算音频", "高端"]),
        Product("H003", "索尼 WH-1000XM5 头戴式降噪耳机 黑色", ProductCategory.HEADPHONE, "索尼", 2499, 2999, 4.7, 140000, 450, "双芯降噪，30小时续航，PD快充，多点连接", ["降噪之王", "头戴式", "长续航"]),
        Product("H004", "索尼 WF-1000XM5 真无线降噪耳机", ProductCategory.HEADPHONE, "索尼", 1699, 1999, 4.6, 95000, 380, "双芯降噪，8小时+16小时续航，IPX4防水", ["降噪", "真无线", "防水"]),
        Product("H005", "华为 FreeBuds Pro 3 雅川青", ProductCategory.HEADPHONE, "华为", 1299, 1499, 4.5, 120000, 550, "麒麟A2芯片，星闪连接，智慧动态降噪，1.5Mbps无损", ["星闪", "无损音质", "华为生态"]),
        Product("H006", "Bose QuietComfort Ultra 头戴式", ProductCategory.HEADPHONE, "Bose", 2999, 3399, 4.6, 55000, 200, "CustomTune智能耳内音场调校，24小时续航，沉浸式音频", ["降噪", "沉浸音频", "舒适"]),
        Product("H007", "小米 Buds 4 Pro 真无线降噪耳机", ProductCategory.HEADPHONE, "小米", 899, 1099, 4.3, 180000, 800, "48dB自适应降噪，独立空间音频，38小时续航", ["性价比", "降噪", "空间音频"]),
        Product("W001", "Apple Watch Series 9 GPS 45mm 星光色", ProductCategory.WATCH, "Apple", 3199, 3499, 4.8, 160000, 600, "S9 SiP芯片，双指互点两下，亮度最高2000尼特", ["健康监测", "苹果生态", "双指互点"]),
        Product("W002", "Apple Watch Ultra 2 49mm 钛金属", ProductCategory.WATCH, "Apple", 6499, 6999, 4.9, 45000, 200, "S9 SiP，3000尼特亮度，36小时续航，双频GPS", ["户外", "专业", "长续航"]),
        Product("W003", "华为 WATCH GT 4 46mm 曜石黑", ProductCategory.WATCH, "华为", 1488, 1688, 4.7, 200000, 900, "14天超长续航，减脂塑形，心律失常提示", ["长续航", "健康", "性价比"]),
        Product("W004", "华为 WATCH 4 Pro 太空探索", ProductCategory.WATCH, "华为", 3999, 4499, 4.6, 60000, 250, "eSIM独立通话，一键微体检，钛合金表壳", ["独立通话", "微体检", "eSIM"]),
        Product("W005", "小米手表S3 47mm 黑色", ProductCategory.WATCH, "小米", 799, 999, 4.4, 140000, 700, "百变表圈，HyperOS，eSIM可选，12天续航", ["性价比", "可换表圈", "eSIM"]),
        Product("W006", "三星 Galaxy Watch6 Classic 47mm", ProductCategory.WATCH, "三星", 2399, 2699, 4.5, 35000, 180, "旋转表圈，身体成分分析，跌倒检测", ["经典设计", "身体成分", "旋转表圈"]),
        Product("C001", "索尼 A7M4 全画幅微单相机 单机身", ProductCategory.CAMERA, "索尼", 16999, 17999, 4.9, 55000, 150, "3300万像素，4K 60p，10-bit 4:2:2，实时眼部对焦", ["全画幅", "视频", "专业"]),
        Product("C002", "佳能 EOS R6 Mark II 全画幅专微", ProductCategory.CAMERA, "佳能", 16499, 17799, 4.8, 42000, 120, "2420万像素，40张/秒连拍，4K 60p，8级防抖", ["全画幅", "高速连拍", "防抖"]),
        Product("C003", "富士 X-T5 微单相机 银色", ProductCategory.CAMERA, "富士", 11990, 12990, 4.7, 38000, 100, "4020万像素，7档五轴防抖，经典拨盘操控", ["复古", "胶片模拟", "高像素"]),
        Product("C004", "大疆 Pocket 3 全能套装", ProductCategory.CAMERA, "大疆", 4499, 4999, 4.8, 220000, 800, "1英寸CMOS，2英寸旋转屏，三轴云台，4K 120p", ["vlog", "云台", "便携"]),
        Product("C005", "GoPro HERO12 Black 运动相机", ProductCategory.CAMERA, "GoPro", 2598, 2998, 4.5, 85000, 400, "5.3K 60p，HyperSmooth 6.0，177度超广角，10米防水", ["运动", "防水", "防抖"]),
        Product("G001", "索尼 PlayStation 5 光驱版", ProductCategory.GAME_CONSOLE, "索尼", 3599, 3899, 4.9, 300000, 600, "4K 120Hz游戏，超高速SSD，触觉反馈手柄，3D音效", ["4K游戏", "独占大作", "高速SSD"]),
        Product("G002", "任天堂 Switch OLED版 红蓝色", ProductCategory.GAME_CONSOLE, "任天堂", 2199, 2599, 4.8, 500000, 1200, "7英寸OLED屏，TV/桌面/掌机三模式，Joy-Con手柄", ["便携", "独占游戏", "家庭"]),
        Product("G003", "微软 Xbox Series X 1TB", ProductCategory.GAME_CONSOLE, "微软", 3899, 4299, 4.7, 120000, 350, "12 Teraflops性能，4K 120Hz，Quick Resume，XGP", ["4K游戏", "XGP", "最强性能"]),
        Product("G004", "Steam Deck OLED 512GB", ProductCategory.GAME_CONSOLE, "Valve", 4599, 5499, 4.6, 85000, 250, "7.4英寸OLED HDR屏，50Wh电池，掌上玩3A大作", ["掌机PC", "Steam", "OLED"]),
        Product("G005", "ROG Ally X 掌机 24GB+1TB", ProductCategory.GAME_CONSOLE, "ROG", 5999, 6499, 4.5, 45000, 180, "AMD Z1 Extreme，80Wh电池，7英寸1080p 120Hz", ["Windows掌机", "高性能", "长续航"]),
    ]


MOCK_PRODUCTS: List[Product] = _products()
_CATEGORIES = sorted(set(p.category.value for p in MOCK_PRODUCTS))


def get_product(pid: str) -> Optional[Product]:
    for p in MOCK_PRODUCTS:
        if p.id == pid:
            return p
    return None
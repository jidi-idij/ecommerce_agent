"""
数据模型层 - 商品模型、状态定义、模拟数据
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import time


class IntentType(str, Enum):
    """用户意图类型"""
    PRODUCT_SEARCH = "product_search"      # 商品搜索
    PRICE_COMPARE = "price_compare"        # 比价
    AFTER_SALES = "after_sales"            # 售后
    RECOMMENDATION = "recommendation"      # 求推荐
    GENERAL = "general"                    # 通用闲聊


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
    id: str                          # 商品ID
    title: str                       # 商品标题
    category: ProductCategory        # 分类
    brand: str                       # 品牌
    price: float                     # 当前价格
    original_price: float            # 原价
    rating: float                    # 评分（0-5）
    sales_count: int                 # 销量
    stock: int                       # 库存
    description: str                 # 商品描述
    tags: List[str] = field(default_factory=list)  # 标签
    image_url: str = ""              # 图片URL
    url: str = ""                    # 商品链接
    
    @property
    def discount_rate(self) -> float:
        """折扣率"""
        if self.original_price <= 0:
            return 1.0
        return round(self.price / self.original_price, 2)
    
    @property
    def is_hot(self) -> bool:
        """是否热门"""
        return self.sales_count > 50000
    
    def to_dict(self) -> Dict[str, Any]:
        """转字典"""
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
            "description": self.description,
            "tags": self.tags,
            "is_hot": self.is_hot,
            "url": self.url,
        }


@dataclass
class UserIntent:
    """用户意图识别结果"""
    intent_type: IntentType            # 意图类型
    keywords: List[str] = field(default_factory=list)  # 关键词
    attributes: Dict[str, Any] = field(default_factory=dict)  # 属性
    raw_query: str = ""                # 原始查询
    confidence: float = 0.0            # 置信度


@dataclass
class SearchResult:
    """搜索结果"""
    products: List[Product] = field(default_factory=list)
    total: int = 0
    search_time_ms: float = 0.0
    source: str = ""  # vector/keyword/hybrid


@dataclass
class SortResult:
    """排序结果"""
    products: List[Product] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class AgentState:
    """
    LangGraph Agent状态
    贯穿整个导购流程的状态对象
    """
    # 输入
    query: str = ""                    # 用户原始查询
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    
    # 意图识别结果
    intent: Optional[UserIntent] = None
    
    # 检索结果
    search_results: List[Product] = field(default_factory=list)
    
    # 排序结果
    sorted_products: List[Product] = field(default_factory=list)
    
    # 生成结果
    response: str = ""                 # 最终回复
    recommended_products: List[Dict] = field(default_factory=list)
    
    # 元数据
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    # 错误处理
    error: Optional[str] = None
    
    # 流式输出
    stream_buffer: str = ""
    is_finished: bool = False

    def log(self, node: str, message: str, data: Any = None):
        """记录执行日志"""
        self.execution_log.append({
            "node": node,
            "message": message,
            "data": data,
            "timestamp": time.time()
        })


# ==================== 模拟商品数据 ====================

def generate_mock_products() -> List[Product]:
    """生成模拟商品数据（50条）"""
    products = [
        # ===== 手机 =====
        Product("P001", "iPhone 15 Pro Max 256GB 钛金属", ProductCategory.PHONE, "Apple",
                9999, 10999, 4.9, 120000, 500,
                "A17 Pro芯片，4800万像素主摄，钛金属边框，支持USB-C",
                ["5G", "旗舰", "拍照", "钛金属"], image_url="https://example.com/p1.jpg"),
        Product("P002", "iPhone 15 Pro 128GB 蓝色钛金属", ProductCategory.PHONE, "Apple",
                7999, 8999, 4.8, 85000, 320,
                "A17 Pro芯片， lighter钛金属设计，专业影像系统",
                ["5G", "旗舰", "轻便"], image_url="https://example.com/p2.jpg"),
        Product("P003", "iPhone 15 128GB 粉色", ProductCategory.PHONE, "Apple",
                5999, 6999, 4.7, 200000, 800,
                "A16芯片，灵动岛设计，4800万像素主摄",
                ["5G", "性价比", "多彩"], image_url="https://example.com/p3.jpg"),
        Product("P004", "华为 Mate 60 Pro 12GB+512GB 雅川青", ProductCategory.PHONE, "华为",
                6999, 6999, 4.9, 300000, 200,
                "麒麟9000S芯片，卫星通话，全焦段超清影像，昆仑玻璃",
                ["5G", "国产", "卫星通话", "商务"], image_url="https://example.com/p4.jpg"),
        Product("P005", "华为 Mate 60 12GB+256GB 白沙银", ProductCategory.PHONE, "华为",
                5499, 5999, 4.8, 150000, 350,
                "麒麟9000S芯片，超可靠玄武架构，全焦段影像",
                ["5G", "国产", "商务"], image_url="https://example.com/p5.jpg"),
        Product("P006", "华为 Pura 70 Ultra 16GB+512GB 星芒黑", ProductCategory.PHONE, "华为",
                8999, 9999, 4.8, 80000, 150,
                "1英寸超聚光伸缩摄像头，超高速风驰闪拍，双卫星通信",
                ["5G", "拍照旗舰", "伸缩镜头"], image_url="https://example.com/p6.jpg"),
        Product("P007", "小米14 Pro 16GB+512GB 岩石青", ProductCategory.PHONE, "小米",
                5499, 5999, 4.7, 180000, 600,
                "骁龙8 Gen3，徕卡影像，2K超色准屏，120W快充",
                ["5G", "性价比", "徕卡"], image_url="https://example.com/p7.jpg"),
        Product("P008", "小米14 12GB+256GB 雪山粉", ProductCategory.PHONE, "小米",
                3999, 4299, 4.6, 250000, 900,
                "骁龙8 Gen3，徕卡光学镜头，4610mAh电池",
                ["5G", "性价比", "小屏旗舰"], image_url="https://example.com/p8.jpg"),
        Product("P009", "Redmi K70 Pro 16GB+512GB 墨羽", ProductCategory.PHONE, "Redmi",
                3299, 3799, 4.5, 220000, 700,
                "骁龙8 Gen3，2K中国屏，120W快充，光影猎人800",
                ["5G", "性价比之王", "游戏"], image_url="https://example.com/p9.jpg"),
        Product("P010", "vivo X100 Pro 16GB+512GB 辰夜黑", ProductCategory.PHONE, "vivo",
                5499, 5999, 4.7, 95000, 280,
                "天玑9300，蔡司APO超级长焦，5400mAh电池",
                ["5G", "拍照", "蔡司"], image_url="https://example.com/p10.jpg"),
        Product("P011", "OPPO Find X7 Ultra 16GB+512GB 海阔天空", ProductCategory.PHONE, "OPPO",
                5999, 6499, 4.7, 65000, 220,
                "双潜望四主摄，哈苏影像，2K钻石屏",
                ["5G", "拍照旗舰", "哈苏"], image_url="https://example.com/p11.jpg"),
        Product("P012", "三星 Galaxy S24 Ultra 12GB+256GB 钛灰", ProductCategory.PHONE, "三星",
                9699, 10699, 4.6, 45000, 180,
                "骁龙8 Gen3 for Galaxy，S Pen，2亿像素，AI功能",
                ["5G", "S Pen", "AI"], image_url="https://example.com/p12.jpg"),
        
        # ===== 笔记本 =====
        Product("L001", "MacBook Pro 14英寸 M3 Pro芯片 18GB+512GB", ProductCategory.LAPTOP, "Apple",
                16999, 18499, 4.9, 60000, 250,
                "M3 Pro芯片，Liquid视网膜XDR显示屏，18小时续航",
                ["M3芯片", "专业", "长续航"], image_url="https://example.com/l1.jpg"),
        Product("L002", "MacBook Air 13.6英寸 M3芯片 16GB+512GB 午夜色", ProductCategory.LAPTOP, "Apple",
                11999, 12999, 4.8, 110000, 400,
                "M3芯片，轻薄无风扇设计，18小时续航，Liquid视网膜屏",
                ["M3芯片", "轻薄", "静音"], image_url="https://example.com/l2.jpg"),
        Product("L003", "华为 MateBook X Pro 2024 微绒典藏版", ProductCategory.LAPTOP, "华为",
                11999, 12999, 4.7, 35000, 150,
                "980g超轻薄，OLED原色屏，酷睿Ultra 9，140W快充",
                ["超轻薄", "OLED", "商务"], image_url="https://example.com/l3.jpg"),
        Product("L004", "联想 ThinkPad X1 Carbon 2024 14英寸", ProductCategory.LAPTOP, "联想",
                14999, 16499, 4.6, 42000, 200,
                "酷睿Ultra 7，2.8K OLED屏，1.1kg，军工品质",
                ["商务", "轻薄", "军工"], image_url="https://example.com/l4.jpg"),
        Product("L005", "联想 小新Pro16 2024 酷睿Ultra 5", ProductCategory.LAPTOP, "联想",
                5999, 6499, 4.5, 160000, 550,
                "酷睿Ultra 5，2.5K 120Hz屏，84Wh电池，大满贯接口",
                ["性价比", "大屏", "学生"], image_url="https://example.com/l5.jpg"),
        Product("L006", "小米 RedmiBook Pro 16 2024", ProductCategory.LAPTOP, "小米",
                4999, 5499, 4.4, 130000, 480,
                "酷睿Ultra 5，3.1K 165Hz屏，99Wh电池",
                ["性价比", "大屏", "高刷"], image_url="https://example.com/l6.jpg"),
        Product("L007", "戴尔 XPS 13 Plus 13.4英寸 OLED触控屏", ProductCategory.LAPTOP, "戴尔",
                12999, 14999, 4.5, 28000, 120,
                "酷睿i7，3.5K OLED触控，无缝键盘，极简设计",
                ["设计", "触控", "轻薄"], image_url="https://example.com/l7.jpg"),
        Product("L008", "华硕灵耀14 2024 酷睿Ultra 9", ProductCategory.LAPTOP, "华硕",
                6999, 7499, 4.6, 75000, 300,
                "酷睿Ultra 9，2.8K OLED屏，75Wh电池，1.19kg",
                ["OLED", "轻薄", "高性能"], image_url="https://example.com/l8.jpg"),
        
        # ===== 平板 =====
        Product("T001", "iPad Pro 11英寸 M4芯片 256GB", ProductCategory.TABLET, "Apple",
                8999, 9499, 4.9, 90000, 380,
                "M4芯片，超精视网膜XDR屏，Apple Pencil Pro",
                ["M4芯片", "专业创作", "Apple Pencil"], image_url="https://example.com/t1.jpg"),
        Product("T002", "iPad Air 6 M2芯片 128GB 蓝色", ProductCategory.TABLET, "Apple",
                4799, 4999, 4.7, 140000, 600,
                "M2芯片，11英寸Liquid视网膜屏，支持Apple Pencil Pro",
                ["M2芯片", "性价比", "学习"], image_url="https://example.com/t2.jpg"),
        Product("T003", "华为 MatePad Pro 13.2英寸 144Hz OLED", ProductCategory.TABLET, "华为",
                5699, 6199, 4.8, 110000, 450,
                "麒麟9000S，144Hz OLED，星闪手写笔，PC级应用",
                ["OLED", "星闪", "办公"], image_url="https://example.com/t3.jpg"),
        Product("T004", "小米平板6S Pro 12.4英寸 骁龙8 Gen2", ProductCategory.TABLET, "小米",
                3299, 3599, 4.5, 85000, 520,
                "骁龙8 Gen2，3K 144Hz屏，120W快充，PC级WPS",
                ["性价比", "大屏", "快充"], image_url="https://example.com/t4.jpg"),
        Product("T005", "三星 Galaxy Tab S9 Ultra 14.6英寸", ProductCategory.TABLET, "三星",
                7999, 8999, 4.6, 25000, 100,
                "骁龙8 Gen2，14.6英寸AMOLED，IP68防水，S Pen",
                ["超大屏", "AMOLED", "S Pen"], image_url="https://example.com/t5.jpg"),
        
        # ===== 耳机 =====
        Product("H001", "AirPods Pro 2 (USB-C) 主动降噪", ProductCategory.HEADPHONE, "Apple",
                1899, 1999, 4.8, 300000, 1000,
                "H2芯片，自适应音频，USB-C充电，单次6小时续航",
                ["降噪", "苹果生态", "便携"], image_url="https://example.com/h1.jpg"),
        Product("H002", "AirPods Max 银色 头戴式", ProductCategory.HEADPHONE, "Apple",
                3999, 4399, 4.7, 80000, 300,
                "H1芯片，计算音频， knit网面穹顶，20小时续航",
                ["头戴式", "计算音频", "高端"], image_url="https://example.com/h2.jpg"),
        Product("H003", "索尼 WH-1000XM5 头戴式降噪耳机 黑色", ProductCategory.HEADPHONE, "索尼",
                2499, 2999, 4.7, 140000, 450,
                "双芯降噪，30小时续航，PD快充，多点连接",
                ["降噪之王", "头戴式", "长续航"], image_url="https://example.com/h3.jpg"),
        Product("H004", "索尼 WF-1000XM5 真无线降噪耳机", ProductCategory.HEADPHONE, "索尼",
                1699, 1999, 4.6, 95000, 380,
                "双芯降噪，8小时+16小时续航，IPX4防水",
                ["降噪", "真无线", "防水"], image_url="https://example.com/h4.jpg"),
        Product("H005", "华为 FreeBuds Pro 3 雅川青", ProductCategory.HEADPHONE, "华为",
                1299, 1499, 4.5, 120000, 550,
                "麒麟A2芯片，星闪连接，智慧动态降噪，1.5Mbps无损",
                ["星闪", "无损音质", "华为生态"], image_url="https://example.com/h5.jpg"),
        Product("H006", "Bose QuietComfort Ultra 头戴式", ProductCategory.HEADPHONE, "Bose",
                2999, 3399, 4.6, 55000, 200,
                "CustomTune智能耳内音场调校，24小时续航，沉浸式音频",
                ["降噪", "沉浸音频", "舒适"], image_url="https://example.com/h6.jpg"),
        Product("H007", "小米 Buds 4 Pro 真无线降噪耳机", ProductCategory.HEADPHONE, "小米",
                899, 1099, 4.3, 180000, 800,
                "48dB自适应降噪，独立空间音频，38小时续航",
                ["性价比", "降噪", "空间音频"], image_url="https://example.com/h7.jpg"),
        
        # ===== 智能手表 =====
        Product("W001", "Apple Watch Series 9 GPS 45mm 星光色", ProductCategory.WATCH, "Apple",
                3199, 3499, 4.8, 160000, 600,
                "S9 SiP芯片，双指互点两下，亮度最高2000尼特",
                ["健康监测", "苹果生态", "双指互点"], image_url="https://example.com/w1.jpg"),
        Product("W002", "Apple Watch Ultra 2 49mm 钛金属", ProductCategory.WATCH, "Apple",
                6499, 6999, 4.9, 45000, 200,
                "S9 SiP，3000尼特亮度，36小时续航，双频GPS",
                ["户外", "专业", "长续航"], image_url="https://example.com/w2.jpg"),
        Product("W003", "华为 WATCH GT 4 46mm 曜石黑", ProductCategory.WATCH, "华为",
                1488, 1688, 4.7, 200000, 900,
                "14天超长续航，减脂塑形，心律失常提示",
                ["长续航", "健康", "性价比"], image_url="https://example.com/w3.jpg"),
        Product("W004", "华为 WATCH 4 Pro 太空探索", ProductCategory.WATCH, "华为",
                3999, 4499, 4.6, 60000, 250,
                "eSIM独立通话，一键微体检，钛合金表壳",
                ["独立通话", "微体检", "eSIM"], image_url="https://example.com/w4.jpg"),
        Product("W005", "小米手表S3 47mm 黑色", ProductCategory.WATCH, "小米",
                799, 999, 4.4, 140000, 700,
                "百变表圈，HyperOS，eSIM可选，12天续航",
                ["性价比", "可换表圈", "eSIM"], image_url="https://example.com/w5.jpg"),
        Product("W006", "三星 Galaxy Watch6 Classic 47mm", ProductCategory.WATCH, "三星",
                2399, 2699, 4.5, 35000, 180,
                "旋转表圈，身体成分分析，跌倒检测",
                ["经典设计", "身体成分", "旋转表圈"], image_url="https://example.com/w6.jpg"),
        
        # ===== 相机 =====
        Product("C001", "索尼 A7M4 全画幅微单相机 单机身", ProductCategory.CAMERA, "索尼",
                16999, 17999, 4.9, 55000, 150,
                "3300万像素，4K 60p，10-bit 4:2:2，实时眼部对焦",
                ["全画幅", "视频", "专业"], image_url="https://example.com/c1.jpg"),
        Product("C002", "佳能 EOS R6 Mark II 全画幅专微", ProductCategory.CAMERA, "佳能",
                16499, 17799, 4.8, 42000, 120,
                "2420万像素，40张/秒连拍，4K 60p，8级防抖",
                ["全画幅", "高速连拍", "防抖"], image_url="https://example.com/c2.jpg"),
        Product("C003", "富士 X-T5 微单相机 银色", ProductCategory.CAMERA, "富士",
                11990, 12990, 4.7, 38000, 100,
                "4020万像素，7档五轴防抖，经典拨盘操控",
                ["复古", "胶片模拟", "高像素"], image_url="https://example.com/c3.jpg"),
        Product("C004", "大疆 Pocket 3 全能套装", ProductCategory.CAMERA, "大疆",
                4499, 4999, 4.8, 220000, 800,
                "1英寸CMOS，2英寸旋转屏，三轴云台，4K 120p",
                ["vlog", "云台", "便携"], image_url="https://example.com/c4.jpg"),
        Product("C005", "GoPro HERO12 Black 运动相机", ProductCategory.CAMERA, "GoPro",
                2598, 2998, 4.5, 85000, 400,
                "5.3K 60p，HyperSmooth 6.0，177度超广角，10米防水",
                ["运动", "防水", "防抖"], image_url="https://example.com/c5.jpg"),
        
        # ===== 游戏机 =====
        Product("G001", "索尼 PlayStation 5 光驱版", ProductCategory.GAME_CONSOLE, "索尼",
                3599, 3899, 4.9, 300000, 600,
                "4K 120Hz游戏，超高速SSD，触觉反馈手柄，3D音效",
                ["4K游戏", "独占大作", "高速SSD"], image_url="https://example.com/g1.jpg"),
        Product("G002", "任天堂 Switch OLED版 红蓝色", ProductCategory.GAME_CONSOLE, "任天堂",
                2199, 2599, 4.8, 500000, 1200,
                "7英寸OLED屏，TV/桌面/掌机三模式，Joy-Con手柄",
                ["便携", "独占游戏", "家庭"], image_url="https://example.com/g2.jpg"),
        Product("G003", "微软 Xbox Series X 1TB", ProductCategory.GAME_CONSOLE, "微软",
                3899, 4299, 4.7, 120000, 350,
                "12 Teraflops性能，4K 120Hz，Quick Resume，XGP",
                ["4K游戏", "XGP", "最强性能"], image_url="https://example.com/g3.jpg"),
        Product("G004", "Steam Deck OLED 512GB", ProductCategory.GAME_CONSOLE, "Valve",
                4599, 5499, 4.6, 85000, 250,
                "7.4英寸OLED HDR屏，50Wh电池，掌上玩3A大作",
                ["掌机PC", "Steam", "OLED"], image_url="https://example.com/g4.jpg"),
        Product("G005", "ROG Ally X 掌机 24GB+1TB", ProductCategory.GAME_CONSOLE, "ROG",
                5999, 6499, 4.5, 45000, 180,
                "AMD Z1 Extreme，80Wh电池，7英寸1080p 120Hz",
                ["Windows掌机", "高性能", "长续航"], image_url="https://example.com/g5.jpg"),
    ]
    return products


# 全局商品数据
MOCK_PRODUCTS: List[Product] = generate_mock_products()


def get_product_by_id(product_id: str) -> Optional[Product]:
    """通过ID获取商品"""
    for p in MOCK_PRODUCTS:
        if p.id == product_id:
            return p
    return None


# 停用词 - 需要从查询中过滤的无意义词
_STOP_WORDS = {
    "推荐", "一款", "一个", "的", "了", "在", "和", "是", "我", "想", "买", "要",
    "请", "给", "有", "什么", "吗", "呢", "吧", "啊", "一下", "看看", "有没有",
    "帮忙", "帮我", "需要", "想要", "准备", "打算", "左右", "大概", "差不多",
    "样子", "那种", "这样", "最好", "比较", "挺", "很", "非常", "特别", "真",
    "有没有", "能否", "可以", "适合", "怎么样", "如何", "哪个", "哪些", "啥",
    "差不多", "上下", "以内", "以内", "元", "块", "钱",
}

# 分类同义词映射
_CATEGORY_ALIASES = {
    "手机": ["手机", "电话", "智能机", "iPhone", "华为", "小米", "安卓"],
    "笔记本": ["笔记本", "电脑", "手提", "MacBook", "ThinkPad", "轻薄本", "游戏本"],
    "平板": ["平板", "iPad", "Pad"],
    "耳机": ["耳机", "耳麦", "耳塞", "蓝牙耳机", "降噪", "AirPods"],
    "智能手表": ["手表", "手环", "Watch", "智能穿戴"],
    "相机": ["相机", "摄像机", "微单", "单反", "拍照", "摄影"],
    "游戏机": ["游戏机", "掌机", "PS5", "Switch", "Xbox", "主机"],
}


def _extract_keywords(query: str) -> List[str]:
    """从自然语言查询中提取核心关键词（n-gram切分）"""
    import re
    cleaned = query
    for sw in sorted(_STOP_WORDS, key=len, reverse=True):
        cleaned = cleaned.replace(sw, " ")
    
    keywords = []
    for token in re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', cleaned):
        if len(token) < 2:
            continue
        if re.match(r'^[a-zA-Z0-9]+$', token):
            keywords.append(token.lower())
        elif len(token) <= 4:
            keywords.append(token)
        else:
            # 长中文切 n-gram（2~4字窗口），避免整串匹配失败
            for n in range(2, 5):
                for i in range(len(token) - n + 1):
                    keywords.append(token[i:i + n])
    
    return keywords


def search_by_keyword(keyword: str) -> List[Product]:
    """
    关键词搜索（模拟ES关键词搜索）
    支持自然语言查询，自动提取核心关键词
    """
    results = []
    keyword_lower = keyword.lower().strip()
    
    # 尝试完整匹配
    for p in MOCK_PRODUCTS:
        text = f"{p.title} {p.description} {p.brand} {' '.join(p.tags)} {p.category.value}"
        if keyword_lower in text.lower():
            results.append(p)
    
    # 如果完整匹配没有结果，提取核心关键词再匹配
    if not results:
        core_keywords = _extract_keywords(keyword)
        # 也保留原始查询中较长的词（品牌名等）
        raw_tokens = [t for t in keyword_lower.split() if len(t) >= 1]
        all_keywords = list(dict.fromkeys(core_keywords + raw_tokens))  # 去重保序
        
        if all_keywords:
            for p in MOCK_PRODUCTS:
                text = f"{p.title} {p.description} {p.brand} {' '.join(p.tags)} {p.category.value}"
                text_lower = text.lower()
                # 任意关键词匹配即命中
                for kw in all_keywords:
                    if kw in text_lower:
                        if p not in results:
                            results.append(p)
                        break
    
    return results

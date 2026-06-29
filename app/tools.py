"""
MCP Tools层 - 商品搜索、价格监控、库存查询等工具封装
供各Agent节点按需调用
"""
import time
from typing import List, Optional, Dict, Any

from app.models import Product, MOCK_PRODUCTS, get_product_by_id
from app.cache import cache


class ProductSearchTool:
    """
    商品搜索工具
    支持关键词搜索和分类筛选
    """
    
    @staticmethod
    def search(query: str, category: str = None, brand: str = None, 
               min_price: float = None, max_price: float = None,
               limit: int = 20) -> List[Dict[str, Any]]:
        """
        关键词搜索商品
        
        Args:
            query: 搜索关键词
            category: 分类过滤
            brand: 品牌过滤
            min_price: 最低价格
            max_price: 最高价格
            limit: 返回数量
            
        Returns:
            商品列表（字典格式）
        """
        print(f"[Tool:search] query={query}, category={category}, brand={brand}")
        start_time = time.time()
        
        # 先查缓存
        cache_key = f"{query}_{category}_{brand}_{min_price}_{max_price}"
        cached = cache.get_keyword_result(cache_key)
        if cached:
            print(f"[Tool:search] 缓存命中，返回{len(cached)}条结果")
            products = [get_product_by_id(pid) for pid in cached if get_product_by_id(pid)]
            return [p.to_dict() for p in products[:limit]]
        
        # 大模型关键词 OR 匹配（空格分词，任一词命中即入选）
        terms = [t for t in query.split() if len(t) >= 2]
        results = []
        for p in MOCK_PRODUCTS:
            text = f"{p.title} {p.description} {p.brand} {' '.join(p.tags)}".lower()
            if any(t in text for t in terms):
                results.append(p)
        
        # 分类过滤
        if category:
            results = [p for p in results if category in p.category.value or category in p.title]
        else:
            # 跨分类全库搜索，不做分类过滤
            pass
        
        # 品牌过滤
        if brand:
            results = [p for p in results if brand.lower() in p.brand.lower()]
        
        # 价格过滤
        if min_price is not None:
            results = [p for p in results if p.price >= min_price]
        if max_price is not None:
            results = [p for p in results if p.price <= max_price]
        
                # 兜底：关键词无命中时返回全库（让排序节点按用户意图处理）
        if not results:
            results = list(MOCK_PRODUCTS)
            if category:
                results = [p for p in results if category in p.category.value or category in p.title]
            if brand:
                results = [p for p in results if brand.lower() in p.brand.lower()]
            if min_price is not None:
                results = [p for p in results if p.price >= min_price]
            if max_price is not None:
                results = [p for p in results if p.price <= max_price]
            print(f"[Tool:search] 关键词无命中，兜底返回全库{len(results)}条")
            
        # 写入缓存
        product_ids = [p.id for p in results]
        cache.set_keyword_result(cache_key, product_ids, ttl=1800)
        
        elapsed = (time.time() - start_time) * 1000
        print(f"[Tool:search] 搜索完成，返回{len(results)}条，耗时{elapsed:.1f}ms")
        
        return [p.to_dict() for p in results[:limit]]
    
    @staticmethod
    def get_product_detail(product_id: str) -> Optional[Dict[str, Any]]:
        """获取商品详情"""
        product = get_product_by_id(product_id)
        if product:
            return product.to_dict()
        return None
    
    @staticmethod
    def get_by_category(category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """按分类获取商品"""
        results = [p for p in MOCK_PRODUCTS if category in p.category.value or category in p.title]
        return [p.to_dict() for p in results[:limit]]
    
    @staticmethod
    def get_hot_products(limit: int = 10) -> List[Dict[str, Any]]:
        """获取热门商品（按销量排序）"""
        # 查缓存
        cached = cache.get_hot_products("all", limit)
        if cached:
            return cached
        
        sorted_products = sorted(MOCK_PRODUCTS, key=lambda p: p.sales_count, reverse=True)
        result = [p.to_dict() for p in sorted_products[:limit]]
        cache.set_hot_products("all", result, limit, ttl=3600)
        return result
    
    @staticmethod
    def get_recommendations(product_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """获取相关推荐"""
        product = get_product_by_id(product_id)
        if not product:
            return []
        
        # 同分类、同品牌优先
        candidates = [p for p in MOCK_PRODUCTS if p.id != product_id]
        
        def score(p: Product) -> float:
            s = 0
            if p.category == product.category:
                s += 5
            if p.brand == product.brand:
                s += 3
            # 价格相近
            price_diff = abs(p.price - product.price) / max(product.price, 1)
            s += max(0, 2 - price_diff)
            return s
        
        candidates.sort(key=score, reverse=True)
        return [p.to_dict() for p in candidates[:limit]]


class PriceMonitorTool:
    """
    价格监控工具
    监控商品价格变动，发现降价商品
    """
    
    @staticmethod
    def get_current_price(product_id: str) -> Optional[float]:
        """获取商品当前价格"""
        product = get_product_by_id(product_id)
        if product:
            return product.price
        return None
    
    @staticmethod
    def get_price_history(product_id: str) -> List[Dict]:
        """获取价格历史"""
        history = cache.get_price_history(product_id)
        if not history:
            # 生成模拟历史数据
            product = get_product_by_id(product_id)
            if product:
                history = PriceMonitorTool._generate_mock_history(product)
        return history
    
    @staticmethod
    def check_price_drop(product_id: str, threshold: float = 0.05) -> Optional[Dict]:
        """
        检查是否降价
        
        Args:
            product_id: 商品ID
            threshold: 降价阈值（默认5%）
            
        Returns:
            降价信息，未降价返回None
        """
        product = get_product_by_id(product_id)
        if not product:
            return None
        
        history = PriceMonitorTool.get_price_history(product_id)
        if len(history) < 2:
            return None
        
        # 计算价格变化
        current = product.price
        original = product.original_price
        
        if original <= 0:
            return None
        
        drop_rate = (original - current) / original
        
        if drop_rate >= threshold:
            return {
                "product_id": product_id,
                "title": product.title,
                "original_price": original,
                "current_price": current,
                "drop_amount": round(original - current, 2),
                "drop_rate": round(drop_rate * 100, 1),
                "is_deal": drop_rate >= 0.15,  # 15%以上为超值
            }
        return None
    
    @staticmethod
    def get_best_deals(limit: int = 10) -> List[Dict]:
        """获取最佳优惠商品"""
        deals = []
        for product in MOCK_PRODUCTS:
            if product.original_price > product.price:
                drop_rate = (product.original_price - product.price) / product.original_price
                deals.append({
                    "product": product.to_dict(),
                    "drop_rate": round(drop_rate * 100, 1),
                    "drop_amount": round(product.original_price - product.price, 2),
                })
        
        deals.sort(key=lambda x: x["drop_rate"], reverse=True)
        return deals[:limit]
    
    @staticmethod
    def compare_prices(product_ids: List[str]) -> List[Dict]:
        """比价多个商品"""
        results = []
        for pid in product_ids:
            product = get_product_by_id(pid)
            if product:
                results.append({
                    "id": product.id,
                    "title": product.title,
                    "price": product.price,
                    "original_price": product.original_price,
                    "discount": f"{product.discount_rate * 100:.0f}%",
                    "brand": product.brand,
                    "rating": product.rating,
                })
        return results
    
    @staticmethod
    def _generate_mock_history(product: Product) -> List[Dict]:
        """生成模拟价格历史"""
        import random
        history = []
        base_price = product.original_price
        for i in range(30):
            day_price = base_price * (1 - i * 0.01 * random.uniform(0.5, 1.5))
            day_price = max(day_price, product.price)
            history.append({
                "price": round(day_price, 2),
                "time": time.strftime("%Y-%m-%d", time.localtime(time.time() - i * 86400))
            })
        return history[::-1]  # 时间正序


class InventoryTool:
    """
    库存查询工具
    查询商品库存状态
    """
    
    @staticmethod
    def check_stock(product_id: str) -> Dict[str, Any]:
        """查询商品库存"""
        product = get_product_by_id(product_id)
        if not product:
            return {"error": "商品不存在"}
        
        status = "充足" if product.stock > 200 else "紧张" if product.stock > 50 else "告急"
        
        return {
            "product_id": product_id,
            "title": product.title,
            "stock": product.stock,
            "status": status,
            "can_buy": product.stock > 0,
        }
    
    @staticmethod
    def check_multi_stock(product_ids: List[str]) -> List[Dict]:
        """批量查询库存"""
        return [InventoryTool.check_stock(pid) for pid in product_ids if get_product_by_id(pid)]
    
    @staticmethod
    def get_stock_alert(threshold: int = 50) -> List[Dict]:
        """获取库存预警商品"""
        alerts = []
        for p in MOCK_PRODUCTS:
            if p.stock <= threshold:
                alerts.append({
                    "product_id": p.id,
                    "title": p.title,
                    "stock": p.stock,
                    "status": "告急" if p.stock < 50 else "紧张",
                })
        return alerts


class ReviewTool:
    """
    评价分析工具
    分析商品评价，提取优缺点
    """
    
    @staticmethod
    def get_product_reviews(product_id: str, limit: int = 5) -> List[Dict]:
        """获取商品评价摘要"""
        product = get_product_by_id(product_id)
        if not product:
            return []
        
        # 模拟评价数据
        mock_reviews = {
            "P001": [
                {"rating": 5, "content": "拍照效果惊艳，钛金属质感很好", "tag": "拍照"},
                {"rating": 5, "content": "A17 Pro性能很强，游戏不卡顿", "tag": "性能"},
                {"rating": 4, "content": "价格偏高，但物有所值", "tag": "价格"},
            ],
            "P004": [
                {"rating": 5, "content": "卫星通话功能很实用，信号很好", "tag": "信号"},
                {"rating": 5, "content": "支持国产，体验不输苹果", "tag": "国产"},
                {"rating": 4, "content": "续航还可以，充电速度快", "tag": "续航"},
            ],
        }
        
        reviews = mock_reviews.get(product_id, [
            {"rating": 5, "content": "质量很好，物流快", "tag": "综合"},
            {"rating": 4, "content": "性价比不错，推荐购买", "tag": "性价比"},
            {"rating": 5, "content": "使用体验超出预期", "tag": "体验"},
        ])
        
        return reviews[:limit]
    
    @staticmethod
    def analyze_pros_cons(product_id: str) -> Dict[str, List[str]]:
        """分析商品优缺点"""
        reviews = ReviewTool.get_product_reviews(product_id)
        
        # 模拟分析结果
        product = get_product_by_id(product_id)
        pros = []
        cons = []
        
        if product:
            if product.rating >= 4.7:
                pros.append("用户评分高，口碑优秀")
            if product.sales_count > 100000:
                pros.append("销量火爆，大众认可")
            if product.discount_rate < 0.9:
                pros.append("当前有优惠，性价比高")
            if "旗舰" in str(product.tags):
                pros.append("旗舰配置，性能强劲")
            
            if product.price > 5000:
                cons.append("价格较高，预算需充足")
            if product.stock < 200:
                cons.append("库存紧张，建议尽快下单")
        
        return {
            "pros": pros if pros else ["用户反馈良好"],
            "cons": cons if cons else ["暂无明显缺点"],
        }


# ==================== 工具注册表 ====================

TOOLS = {
    "product_search": ProductSearchTool,
    "price_monitor": PriceMonitorTool,
    "inventory": InventoryTool,
    "review": ReviewTool,
}


def get_tool(tool_name: str):
    """获取工具类"""
    return TOOLS.get(tool_name)


def list_tools() -> List[str]:
    """列出所有可用工具"""
    return list(TOOLS.keys())

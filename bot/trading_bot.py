"""
Main trading bot implementation
"""
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import logging

from core import ProjectFWOptimizer, Market, PortfolioConstraints, OptimizationStatus
from data import PolymarketAPI
from strategies import ProbabilityModel, SimpleEdgeModel

logger = logging.getLogger(__name__)

class PolymarketTradingBot:
    """
    Complete trading bot using ProjectFW + Kelly Criterion
    
    Usage:
        bot = PolymarketTradingBot(capital=10000, paper_mode=True)
        bot.run()  # Single cycle
        bot.start(interval_minutes=60)  # Continuous
    """
    
    def __init__(self,
                 capital: float = 10000.0,
                 constraints: Optional[PortfolioConstraints] = None,
                 model: Optional[ProbabilityModel] = None,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 paper_mode: bool = True):
        
        self.capital = capital
        self.constraints = constraints or PortfolioConstraints()
        self.model = model or SimpleEdgeModel()
        self.optimizer = ProjectFWOptimizer()
        self.api = PolymarketAPI()
        
        self.paper_mode = paper_mode
        self.api_key = api_key
        self.api_secret = api_secret
        
        self.trade_history = []
        self.performance_log = []
    
    def fetch_markets(self, min_edge: float = 0.03) -> List[Market]:
        """
        Fetch and filter markets with edge
        
        Args:
            min_edge: Minimum edge required to include market
            
        Returns:
            List of Market objects
        """
        logger.info("Fetching markets from Polymarket...")
        raw_markets = self.api.get_active_markets(limit=50)
        markets = []
        
        for raw in raw_markets:
            try:
                tokens = raw.get("tokens", [])
                if len(tokens) < 2:
                    continue
                
                token_yes = tokens[0].get("token_id", "")
                token_no = tokens[1].get("token_id", "")
                
                price = self.api.get_price(token_yes)
                if price <= 0.05 or price >= 0.95:
                    continue
                
                your_prob = self.model.estimate_probability(raw)
                edge = abs(your_prob - price)
                
                if edge < min_edge:
                    continue
                
                market = Market(
                    condition_id=raw.get("conditionId", ""),
                    question=raw.get("question", "")[:60],
                    token_id_yes=token_yes,
                    token_id_no=token_no,
                    current_price=price,
                    your_probability=your_prob,
                    liquidity=float(raw.get("liquidity", 0)),
                    volume_24h=float(raw.get("volume", 0)),
                    category=raw.get("category", "General"),
                    resolution_date=raw.get("resolutionDate", "")
                )
                
                markets.append(market)
                
            except Exception as e:
                logger.error(f"Error processing market: {e}")
                continue
        
        logger.info(f"Selected {len(markets)} markets with edge > {min_edge:.1%}")
        return markets
    
    def optimize_portfolio(self, markets: List[Market]) -> np.ndarray:
        """Run Kelly optimization"""
        if not markets:
            return np.array([])
        
        allocations, status, info = self.optimizer.optimize(markets, self.constraints)
        
        logger.info(f"Optimization {status.value} in {info['iterations']} iterations")
        logger.info(f"Expected log utility: {info['final_objective']:.6f}")
        
        return allocations
    
    def generate_orders(self, markets: List[Market], 
                       allocations: np.ndarray) -> List[Dict]:
        """Convert allocations to orders"""
        orders = []
        
        for i, m in enumerate(markets):
            alloc = allocations[i]
            if alloc < 0.001:
                continue
            
            amount = alloc * self.capital
            
            direction = "YES" if m.edge > 0 else "NO"
            token_id = m.token_id_yes if m.edge > 0 else m.token_id_no
            
            order = {
                "market": m.question,
                "condition_id": m.condition_id,
                "token_id": token_id,
                "direction": direction,
                "size": amount,
                "allocation": alloc,
                "market_price": m.current_price,
                "your_prob": m.your_probability,
                "edge": abs(m.edge),
                "category": m.category,
                "timestamp": datetime.now().isoformat()
            }
            
            orders.append(order)
        
        return orders
    
    def execute_paper_trades(self, orders: List[Dict]):
        """Simulate trade execution"""
        print("\n" + "=" * 80)
        print("📊 PAPER TRADE EXECUTION")
        print("=" * 80)
        
        total = 0
        for order in orders:
            print(f"\n🎯 {order['market']}")
            print(f"   Direction: BUY {order['direction']}")
            print(f"   Size: ${order['size']:,.2f} ({order['allocation']:.1%})")
            print(f"   Market Price: {order['market_price']:.1%}")
            print(f"   Your Estimate: {order['your_prob']:.1%}")
            print(f"   Edge: {order['edge']:.1%}")
            total += order['size']
            self.trade_history.append(order)
        
        print(f"\n{'=' * 80}")
        print(f"💰 TOTAL: ${total:,.2f} ({total/self.capital:.1%})")
        print(f"💵 CASH: ${self.capital - total:,.2f}")
        print("=" * 80)
    
    def run(self):
        """Execute one trading cycle"""
        print(f"\n{'=' * 80}")
        print(f"🚀 POLYMARKET BOT - {datetime.now()}")
        print(f"Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        print(f"Capital: ${self.capital:,.2f}")
        print(f"{'=' * 80}\n")
        
        try:
            markets = self.fetch_markets()
            if not markets:
                print("No tradeable markets found")
                return
            
            allocations = self.optimize_portfolio(markets)
            orders = self.generate_orders(markets, allocations)
            
            if not orders:
                print("No orders generated")
                return
            
            if self.paper_mode:
                self.execute_paper_trades(orders)
            else:
                logger.info("Live execution not implemented in this version")
            
        except Exception as e:
            logger.error(f"Trading cycle failed: {e}", exc_info=True)
    
    def start(self, interval_minutes: int = 60):
        """Start continuous trading"""
        import time
        
        print(f"\nStarting bot (interval: {interval_minutes} min)")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.run()
                print(f"\nSleeping {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\n\nBot stopped")
"""
Main trading bot implementation
"""
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import logging

from core import ProjectFWOptimizer, Market, PortfolioConstraints, OptimizationStatus
from data import PolymarketAPI
from strategies import ProbabilityModel, SimpleEdgeModel, YesNoArbScanner

logger = logging.getLogger(__name__)

class PolymarketTradingBot:
    """
    Complete trading bot using ProjectFW + Kelly Criterion
    """
    def __init__(self,
                 capital: float = 10000.0,
                 constraints: Optional[PortfolioConstraints] = None,
                 model: Optional[ProbabilityModel] = None,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 paper_mode: bool = True,
                 enable_yes_no_arb: bool = True,
                 arb_fee_buffer: float = 0.02,
                 arb_max_per_market: float = 0.05):
        
        self.capital = capital
        self.constraints = constraints or PortfolioConstraints()
        self.model = model or SimpleEdgeModel()
        self.optimizer = ProjectFWOptimizer()
        self.api = PolymarketAPI()
        
        self.paper_mode = paper_mode
        self.api_key = api_key
        self.api_secret = api_secret

        # Yes/No arbitrage
        self.enable_yes_no_arb = enable_yes_no_arb
        self.arb_scanner = YesNoArbScanner(
            fee_buffer=arb_fee_buffer,
            max_per_market=arb_max_per_market
        )
        
        self.trade_history = []
        self.performance_log = []
    
    def fetch_markets(self, min_edge: float = 0.03) -> List[Market]:
        logger.info("Fetching markets from Polymarket...")
        raw_markets = self.api.get_active_markets(limit=50)
        markets = []

        for raw in raw_markets:
            try:
                # Gamma API currently provides token ids via `clobTokenIds` and prices via `outcomePrices`.
                token_ids = raw.get("clobTokenIds") or []
                prices = raw.get("outcomePrices") or []

                # Gamma sometimes returns these as JSON-encoded strings.
                import json
                if isinstance(token_ids, str):
                    token_ids = json.loads(token_ids)
                if isinstance(prices, str):
                    prices = json.loads(prices)

                if len(token_ids) < 2:
                    continue

                token_yes = str(token_ids[0])
                token_no = str(token_ids[1])

                # Prefer Gamma's outcomePrices[0] (YES) to avoid extra HTTP calls.
                try:
                    price = float(prices[0]) if len(prices) >= 1 else 0.0
                except Exception:
                    price = 0.0

                if price <= 0:
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
                    resolution_date=raw.get("endDateIso", raw.get("endDate", ""))
                )
                
                markets.append(market)
                
            except Exception as e:
                logger.error(f"Error processing market: {e}")
                continue
        
        logger.info(f"Selected {len(markets)} markets with edge > {min_edge:.1%}")
        return markets
    
    def optimize_portfolio(self, markets: List[Market]) -> np.ndarray:
        if not markets:
            return np.array([])
        
        allocations, status, info = self.optimizer.optimize(markets, self.constraints)
        logger.info(f"Optimization {status.value} in {info['iterations']} iterations")
        logger.info(f"Expected log utility: {info['final_objective']:.6f}")
        return allocations
    
    def generate_orders(self, markets: List[Market], allocations: np.ndarray) -> List[Dict]:

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

    def scan_yes_no_arbitrage(self) -> List[Dict]:
        if not self.enable_yes_no_arb:
            return []

        logger.info("Scanning YES/NO sum arbitrage...")
        raw_markets = self.api.get_active_markets(limit=50)
        orders: List[Dict] = []

        for raw in raw_markets:
            try:
                token_ids = raw.get("clobTokenIds") or []
                if isinstance(token_ids, str):
                    import json
                    token_ids = json.loads(token_ids)
                if len(token_ids) < 2:
                    continue

                token_yes = str(token_ids[0])
                token_no = str(token_ids[1])

                book_yes = self.api.get_orderbook(token_yes)
                book_no = self.api.get_orderbook(token_no)

                signal = self.arb_scanner.scan(raw, book_yes, book_no, self.capital)
                if not signal:
                    continue

                order = {
                    "type": "YES_NO_ARB",
                    "market": signal.question,
                    "condition_id": signal.condition_id,
                    "token_id_yes": signal.token_yes,
                    "token_id_no": signal.token_no,
                    "ask_yes": signal.ask_yes,
                    "ask_no": signal.ask_no,
                    "sum_price": signal.sum_price,
                    "edge": signal.edge,
                    "size": signal.size_dollars,
                    "timestamp": datetime.now().isoformat()
                }
                orders.append(order)

            except Exception as e:
                logger.error(f"Arb scan error: {e}")
                continue

        logger.info(f"Found {len(orders)} YES/NO arb opportunities")
        return orders
    
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

            arb_orders = self.scan_yes_no_arbitrage()

            if not orders and not arb_orders:
                print("No orders generated")
                return

            if self.paper_mode:
                if orders:
                    self.execute_paper_trades(orders)

                if arb_orders:
                    print("\n" + "=" * 80)
                    print("🧩 YES/NO ARBITRAGE (PAPER)")
                    print("=" * 80)
                    for o in arb_orders:
                        print(f"\n🎯 {o['market']}")
                        print(f"   Buy YES @ {o['ask_yes']:.3f} + NO @ {o['ask_no']:.3f} = {o['sum_price']:.3f}")
                        print(f"   Edge: {o['edge']:.2%} | Size: ${o['size']:.2f}")
                        self.trade_history.append(o)
                    print("=" * 80)
            else:
                logger.info("Live execution not implemented in this version")
            
        except Exception as e:
            logger.error(f"Trading cycle failed: {e}", exc_info=True)


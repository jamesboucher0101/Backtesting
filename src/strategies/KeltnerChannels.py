from typing import Dict, Any, List
import pandas as pd
import numpy as np
import itertools
from .base import Strategy


class KeltnerChannelsStrategy(Strategy):
    """
    Keltner Channels Strategy - Complete rewrite from Pine Script
    
    Pine Script Reference:
    - length = input.int(20, minval=1)
    - mult = input.float(2.0, "Multiplier")
    - src = input(close, title="Source")
    - exp = input(true, "Use Exponential MA")
    - BandsStyle = input.string("Average True Range", options = ["Average True Range", "True Range", "Range"])
    - atrlength = input(10, "ATR Length")
    """
    
    @property
    def name(self) -> str:
        return "KeltnerChannels"

    def suggest_parameters(self, trial=None) -> Dict[str, Any]:
        return {
            'length': 20,
            'mult': 2.0,
            'src': 'close',
            'exp': True,
            'BandsStyle': 'Average True Range',
            'atrlength': 10
        }

    @classmethod
    def get_parameter_combinations(cls) -> List[Dict[str, Any]]:
        # Generate parameter combinations
        lengths = [10, 15, 20, 25, 30]
        mults = [1.5, 2.0, 2.5, 3.0]
        sources = ['close', 'open', 'high', 'low']
        exps = [True, False]
        band_styles = ['Average True Range', 'True Range', 'Range']
        atr_lengths = [5, 10, 14, 20]
        
        combinations = []
        for length, mult, src, exp, band_style, atr_length in itertools.product(
            lengths, mults, sources, exps, band_styles, atr_lengths
        ):
            combinations.append({
                'length': length,
                'mult': mult,
                'src': src,
                'exp': exp,
                'BandsStyle': band_style,
                'atrlength': atr_length
            })
        
        return combinations

    def prepare(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare data following Pine Script logic exactly:
        
        Pine Script lines:
        121: esma(source, length)=>
        122:     s = ta.sma(source, length)
        123:     e = ta.ema(source, length)
        124:     exp ? e : s
        125: ma = esma(src, length)
        126: rangema = BandsStyle == "True Range" ? ta.tr(true) : BandsStyle == "Average True Range" ? ta.atr(atrlength) : ta.rma(high - low, length)
        127: upper = ma + rangema * mult
        128: lower = ma - rangema * mult
        129: crossUpper = ta.crossover(src, upper)
        130: crossLower = ta.crossunder(src, lower)
        """
        import talib
        
        out = df.copy()
        
        # Extract parameters
        length = int(params.get('length', 20))
        mult = float(params.get('mult', 2.0))
        src_name = params.get('src', 'close')
        exp = params.get('exp', True)
        BandsStyle = params.get('BandsStyle', 'Average True Range')
        atrlength = int(params.get('atrlength', 10))
        
        # Get source data
        if src_name == 'close':
            src = df['close']
        elif src_name == 'open':
            src = df['open']
        elif src_name == 'high':
            src = df['high']
        elif src_name == 'low':
            src = df['low']
        else:
            src = df['close']
        
        # esma function (lines 121-124)
        if exp:
            ma = talib.EMA(src, timeperiod=length)
        else:
            ma = talib.SMA(src, timeperiod=length)
        
        # rangema calculation (line 126)
        if BandsStyle == "True Range":
            rangema = talib.TRANGE(df['high'], df['low'], df['close'])
        elif BandsStyle == "Average True Range":
            rangema = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atrlength)
        else:  # "Range"
            # ta.rma(high - low, length) - RMA of high-low range
            hl_range = df['high'] - df['low']
            # Approximate RMA with EMA (RMA ≈ EMA with alpha=1/length)
            rangema = talib.EMA(hl_range, timeperiod=length)
        
        # Keltner Channels (lines 127-128)
        upper = ma + rangema * mult
        lower = ma - rangema * mult
        
        # Crossover detection (lines 129-130)
        # ta.crossover(src, upper) = src > upper AND src[1] <= upper[1]
        crossUpper = (src > upper) & (src.shift(1) <= upper.shift(1))
        # ta.crossunder(src, lower) = src < lower AND src[1] >= lower[1]
        crossLower = (src < lower) & (src.shift(1) >= lower.shift(1))
        
        # Add to output
        out['src'] = src
        out['ma'] = ma
        out['rangema'] = rangema
        out['upper'] = upper
        out['lower'] = lower
        out['crossUpper'] = crossUpper
        out['crossLower'] = crossLower
        
        # Pine Script lines 131-140: bprice, sprice, crossBcond, crossScond logic
        # Line 131-132: bprice = 0.0, bprice := crossUpper ? high+syminfo.mintick : nz(bprice[1])
        # Line 133-134: sprice = 0.0, sprice := crossLower ? low-syminfo.mintick : nz(sprice[1])
        
        # Initialize
        out['bprice'] = 0.0
        out['sprice'] = 0.0
        out['crossBcond'] = False
        out['crossScond'] = False
        
        # Calculate mintick approximation
        mintick = src.diff().abs().median() * 0.001
        
        # Forward fill bprice and sprice
        for i in range(1, len(out)):
            # bprice logic
            if crossUpper.iloc[i]:
                out.iloc[i, out.columns.get_loc('bprice')] = df['high'].iloc[i] + mintick
            else:
                out.iloc[i, out.columns.get_loc('bprice')] = out['bprice'].iloc[i-1]
            
            # sprice logic  
            if crossLower.iloc[i]:
                out.iloc[i, out.columns.get_loc('sprice')] = df['low'].iloc[i] - mintick
            else:
                out.iloc[i, out.columns.get_loc('sprice')] = out['sprice'].iloc[i-1]
            
            # crossBcond logic (line 135-136)
            if crossUpper.iloc[i]:
                out.iloc[i, out.columns.get_loc('crossBcond')] = True
            else:
                out.iloc[i, out.columns.get_loc('crossBcond')] = out['crossBcond'].iloc[i-1]
            
            # crossScond logic (line 137-138)
            if crossLower.iloc[i]:
                out.iloc[i, out.columns.get_loc('crossScond')] = True
            else:
                out.iloc[i, out.columns.get_loc('crossScond')] = out['crossScond'].iloc[i-1]
        
        # Cancel conditions (lines 139-140)
        # cancelBcond = crossBcond and (src < ma or high >= bprice)
        # cancelScond = crossScond and (src > ma or low <= sprice)
        out['cancelBcond'] = out['crossBcond'] & ((src < ma) | (df['high'] >= out['bprice']))
        out['cancelScond'] = out['crossScond'] & ((src > ma) | (df['low'] <= out['sprice']))
        
        # Reset crossBcond/crossScond when cancelled
        for i in range(len(out)):
            if out['cancelBcond'].iloc[i]:
                out.iloc[i, out.columns.get_loc('crossBcond')] = False
            if out['cancelScond'].iloc[i]:
                out.iloc[i, out.columns.get_loc('crossScond')] = False
        
        return out

    def warmup_period(self, params: Dict[str, Any]) -> int:
        length = int(params.get('length', 20))
        atrlength = int(params.get('atrlength', 10))
        return max(length, atrlength) + 5

    def decide_position(self, df: pd.DataFrame, i: int, prev_position: int, params: Dict[str, Any]) -> int:
        """
        Pine Script entry logic (lines 141-152):
        
        141: if (cancelBcond)
        142:     strategy.cancel("KltChLE")
        143: if (crossUpper)
        144:     if strategy.position_size < 0
        145:         alert(ShortExitMSG.parseing(per = 100,size = math.abs(strategy.position_size)),freq = alert.freq_once_per_bar )
        146:     strategy.entry("KltChLE", strategy.long, stop=bprice, comment="KltChLE", alert_message = LongEntryMSG.parseing( size = currentSize))
        147: if (cancelScond)
        148:     strategy.cancel("KltChSE")
        149: if (crossLower)
        150:     if strategy.position_size > 0
        151:         alert(LongExitMSG.parseing(per = 100,size = math.abs(strategy.position_size)),freq = alert.freq_once_per_bar )
        152:     strategy.entry("KltChSE", strategy.short, stop=sprice, comment="KltChSE", alert_message = ShortEntryMSG.parseing( size = currentSize ))
        """
        
        if i < 2:
            return prev_position
        
        # Get current data
        crossUpper = df['crossUpper'].iloc[i] if 'crossUpper' in df.columns else False
        crossLower = df['crossLower'].iloc[i] if 'crossLower' in df.columns else False
        bprice = df['bprice'].iloc[i] if 'bprice' in df.columns else 0.0
        sprice = df['sprice'].iloc[i] if 'sprice' in df.columns else 0.0
        crossBcond = df['crossBcond'].iloc[i] if 'crossBcond' in df.columns else False
        crossScond = df['crossScond'].iloc[i] if 'crossScond' in df.columns else False
        cancelBcond = df['cancelBcond'].iloc[i] if 'cancelBcond' in df.columns else False
        cancelScond = df['cancelScond'].iloc[i] if 'cancelScond' in df.columns else False
        
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        
        # Check for valid data
        if pd.isna(df['ma'].iloc[i] if 'ma' in df.columns else np.nan):
            return prev_position
        
        # Line 143-146: if (crossUpper)
        if crossUpper:
            # Line 144-145: if strategy.position_size < 0 -> exit short first
            if prev_position < 0:
                return 0  # Exit short position
            # Line 146: strategy.entry("KltChLE", strategy.long, stop=bprice, ...)
            # This sets up a pending long stop order
        
        # Line 149-152: if (crossLower)  
        if crossLower:
            # Line 150-151: if strategy.position_size > 0 -> exit long first
            if prev_position > 0:
                return 0  # Exit long position
            # Line 152: strategy.entry("KltChSE", strategy.short, stop=sprice, ...)
            # This sets up a pending short stop order
        
        # Execute pending stop orders
        # Long stop order: execute when high >= bprice and we have crossBcond and not cancelled
        if (prev_position == 0 and crossBcond and not cancelBcond and 
            bprice > 0 and current_high >= bprice):
            return 1  # Enter long
        
        # Short stop order: execute when low <= sprice and we have crossScond and not cancelled
        if (prev_position == 0 and crossScond and not cancelScond and 
            sprice > 0 and current_low <= sprice):
            return -1  # Enter short
        
        return prev_position

    def get_trade_info(self, df: pd.DataFrame, i: int, position: int, params: Dict[str, Any]) -> Dict[str, Any]:
        if i >= len(df):
            return {}
        
        return {
            'src': df['src'].iloc[i] if 'src' in df.columns and i < len(df) else None,
            'ma': df['ma'].iloc[i] if 'ma' in df.columns and i < len(df) else None,
            'upper': df['upper'].iloc[i] if 'upper' in df.columns and i < len(df) else None,
            'lower': df['lower'].iloc[i] if 'lower' in df.columns and i < len(df) else None,
            'crossUpper': df['crossUpper'].iloc[i] if 'crossUpper' in df.columns and i < len(df) else None,
            'crossLower': df['crossLower'].iloc[i] if 'crossLower' in df.columns and i < len(df) else None,
            'bprice': df['bprice'].iloc[i] if 'bprice' in df.columns and i < len(df) else None,
            'sprice': df['sprice'].iloc[i] if 'sprice' in df.columns and i < len(df) else None,
            'crossBcond': df['crossBcond'].iloc[i] if 'crossBcond' in df.columns and i < len(df) else None,
            'crossScond': df['crossScond'].iloc[i] if 'crossScond' in df.columns and i < len(df) else None,
            'cancelBcond': df['cancelBcond'].iloc[i] if 'cancelBcond' in df.columns and i < len(df) else None,
            'cancelScond': df['cancelScond'].iloc[i] if 'cancelScond' in df.columns and i < len(df) else None,
        }

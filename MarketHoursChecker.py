import datetime as dt
from datetime import datetime, timezone
import pytz
import calendar
from typing import Dict, List, Tuple, Optional
import logging

class MarketHoursChecker:
    """
    Comprehensive market hours and instrument availability checker for forex trading.
    
    This class handles:
    - Forex market session times (Sydney, Tokyo, London, New York)
    - Major holidays affecting forex pairs
    - Broker-specific maintenance windows
    - Instrument availability verification
    """
    
    def __init__(self):
        self.logger = logging.getLogger('MarketHoursChecker')
        
        # Major forex market sessions (UTC times)
        self.market_sessions = {
            'sydney': {'open': 21, 'close': 6},     # 21:00 UTC - 06:00 UTC (next day)
            'tokyo': {'open': 23, 'close': 8},      # 23:00 UTC - 08:00 UTC (next day)
            'london': {'open': 7, 'close': 16},     # 07:00 UTC - 16:00 UTC
            'new_york': {'open': 12, 'close': 21}   # 12:00 UTC - 21:00 UTC
        }
        
        # Currency-specific trading characteristics
        self.currency_characteristics = {
            'AUD': {'primary_sessions': ['sydney', 'tokyo'], 'secondary_sessions': ['london']},
            'NZD': {'primary_sessions': ['sydney', 'tokyo'], 'secondary_sessions': ['london']},
            'JPY': {'primary_sessions': ['tokyo'], 'secondary_sessions': ['london', 'new_york']},
            'EUR': {'primary_sessions': ['london'], 'secondary_sessions': ['new_york']},
            'GBP': {'primary_sessions': ['london'], 'secondary_sessions': ['new_york']},
            'USD': {'primary_sessions': ['new_york'], 'secondary_sessions': ['london']},
            'CAD': {'primary_sessions': ['new_york'], 'secondary_sessions': ['london']},
            'CHF': {'primary_sessions': ['london'], 'secondary_sessions': ['new_york']}
        }
        
        # Major holidays by country (simplified - in practice you'd use a holiday calendar API)
        self.major_holidays = {
            'US': ['2025-01-01', '2025-01-20', '2025-02-17', '2025-05-26', '2025-07-04', '2025-09-01', '2025-10-13', '2025-11-11', '2025-11-27', '2025-12-25'],
            'UK': ['2025-01-01', '2025-04-18', '2025-04-21', '2025-05-05', '2025-05-26', '2025-08-25', '2025-12-25', '2025-12-26'],
            'EU': ['2025-01-01', '2025-04-18', '2025-04-21', '2025-05-01', '2025-12-25', '2025-12-26'],
            'JP': ['2025-01-01', '2025-01-13', '2025-02-11', '2025-02-23', '2025-03-20', '2025-04-29', '2025-05-03', '2025-05-04', '2025-05-05'],
            'AU': ['2025-01-01', '2025-01-27', '2025-04-18', '2025-04-21', '2025-04-25', '2025-06-09', '2025-12-25', '2025-12-26']
        }
        
        # Currency to country mapping
        self.currency_countries = {
            'USD': 'US', 'EUR': 'EU', 'GBP': 'UK', 'JPY': 'JP', 'AUD': 'AU', 'CAD': 'US', 'CHF': 'EU', 'NZD': 'AU'
        }
        
        # FXCM typical maintenance windows (UTC)
        self.broker_maintenance = {
            'daily': {'start': '22:00', 'end': '22:05', 'days': [5]},  # Friday maintenance
            'weekly': {'start': '21:00', 'end': '22:30', 'day': 'sunday'}  # Sunday extended maintenance
        }

    def is_market_open(self, instrument: str, check_time: Optional[datetime] = None) -> Dict:
        """
        Check if the market is open for a specific instrument at a given time.
        
        Args:
            instrument: Currency pair (e.g., 'AUD/USD', 'EUR/GBP')
            check_time: Time to check (default: current UTC time)
            
        Returns:
            Dict with availability status and details
        """
        if check_time is None:
            check_time = datetime.now(timezone.utc)
        elif check_time.tzinfo is None:
            check_time = check_time.replace(tzinfo=timezone.utc)
        
        result = {
            'is_available': False,
            'status': 'unknown',
            'reason': '',
            'next_open': None,
            'current_sessions': [],
            'liquidity_level': 'unknown',
            'instrument': instrument,
            'check_time': check_time.isoformat()
        }
        
        try:
            # Parse currency pair
            if '/' not in instrument:
                result['status'] = 'invalid_instrument'
                result['reason'] = f"Invalid instrument format: {instrument}"
                return result
                
            base_currency, quote_currency = instrument.split('/')
            
            # Check if it's weekend (Saturday or Sunday in UTC)
            weekday = check_time.weekday()  # 0=Monday, 6=Sunday
            if weekday == 5:  # Saturday
                result['status'] = 'weekend_closed'
                result['reason'] = 'Forex market closed on Saturday'
                result['next_open'] = self._get_next_market_open(check_time)
                return result
            elif weekday == 6:  # Sunday
                # Check if it's before Sunday 21:00 UTC (market opening)
                if check_time.hour < 21:
                    result['status'] = 'weekend_closed'
                    result['reason'] = 'Forex market closed - opens Sunday 21:00 UTC'
                    result['next_open'] = check_time.replace(hour=21, minute=0, second=0, microsecond=0)
                    return result
            
            # Check for holidays
            holiday_check = self._check_holidays(base_currency, quote_currency, check_time)
            if holiday_check['has_holiday']:
                result['status'] = 'holiday_closed'
                result['reason'] = holiday_check['reason']
                result['next_open'] = holiday_check['next_open']
                return result
            
            # Check broker maintenance
            maintenance_check = self._check_broker_maintenance(check_time)
            if maintenance_check['is_maintenance']:
                result['status'] = 'maintenance'
                result['reason'] = maintenance_check['reason']
                result['next_open'] = maintenance_check['end_time']
                return result
            
            # Check active trading sessions
            active_sessions = self._get_active_sessions(check_time)
            result['current_sessions'] = active_sessions
            
            # Determine liquidity and availability
            liquidity_info = self._assess_liquidity(base_currency, quote_currency, active_sessions)
            result['liquidity_level'] = liquidity_info['level']
            
            # Market is considered open if there's at least minimal liquidity
            if liquidity_info['level'] in ['high', 'medium', 'low']:
                result['is_available'] = True
                result['status'] = 'open'
                result['reason'] = f"Market open - {liquidity_info['description']}"
            else:
                result['is_available'] = False
                result['status'] = 'low_liquidity'
                result['reason'] = 'Market technically open but very low liquidity'
                result['next_open'] = self._get_next_high_liquidity_time(base_currency, quote_currency, check_time)
            
        except Exception as e:
            result['status'] = 'error'
            result['reason'] = f"Error checking market status: {str(e)}"
            self.logger.error(f"Error in is_market_open: {e}")
        
        return result

    def _get_active_sessions(self, check_time: datetime) -> List[str]:
        """Get list of currently active trading sessions"""
        active_sessions = []
        current_hour = check_time.hour
        
        for session_name, times in self.market_sessions.items():
            open_hour = times['open']
            close_hour = times['close']
            
            # Handle sessions that cross midnight
            if open_hour > close_hour:
                if current_hour >= open_hour or current_hour < close_hour:
                    active_sessions.append(session_name)
            else:
                if open_hour <= current_hour < close_hour:
                    active_sessions.append(session_name)
        
        return active_sessions

    def _assess_liquidity(self, base_currency: str, quote_currency: str, active_sessions: List[str]) -> Dict:
        """Assess liquidity level based on active sessions and currency characteristics"""
        if not active_sessions:
            return {'level': 'very_low', 'description': 'No major sessions active'}
        
        # Get primary sessions for both currencies
        base_primary = self.currency_characteristics.get(base_currency, {}).get('primary_sessions', [])
        quote_primary = self.currency_characteristics.get(quote_currency, {}).get('primary_sessions', [])
        base_secondary = self.currency_characteristics.get(base_currency, {}).get('secondary_sessions', [])
        quote_secondary = self.currency_characteristics.get(quote_currency, {}).get('secondary_sessions', [])
        
        # Calculate liquidity score
        score = 0
        description_parts = []
        
        # Primary sessions get higher score
        for session in active_sessions:
            if session in base_primary or session in quote_primary:
                score += 3
                description_parts.append(f"{session} (primary)")
            elif session in base_secondary or session in quote_secondary:
                score += 2
                description_parts.append(f"{session} (secondary)")
            else:
                score += 1
                description_parts.append(f"{session} (other)")
        
        # Determine liquidity level
        if score >= 5:
            level = 'high'
        elif score >= 3:
            level = 'medium'
        elif score >= 1:
            level = 'low'
        else:
            level = 'very_low'
        
        description = f"Sessions: {', '.join(description_parts)}"
        
        return {'level': level, 'description': description, 'score': score}

    def _check_holidays(self, base_currency: str, quote_currency: str, check_time: datetime) -> Dict:
        """Check if current date is a major holiday for either currency"""
        check_date = check_time.date().isoformat()
        
        base_country = self.currency_countries.get(base_currency)
        quote_country = self.currency_countries.get(quote_currency)
        
        affected_countries = []
        
        if base_country and check_date in self.major_holidays.get(base_country, []):
            affected_countries.append(f"{base_currency} ({base_country})")
        
        if quote_country and check_date in self.major_holidays.get(quote_country, []):
            affected_countries.append(f"{quote_currency} ({quote_country})")
        
        if affected_countries:
            return {
                'has_holiday': True,
                'reason': f"Holiday in {', '.join(affected_countries)}",
                'next_open': self._get_next_trading_day(check_time)
            }
        
        return {'has_holiday': False}

    def _check_broker_maintenance(self, check_time: datetime) -> Dict:
        """Check if broker is in maintenance window"""
        weekday = check_time.weekday()
        current_time = check_time.strftime('%H:%M')
        
        # Check daily maintenance (typically Friday)
        daily_maint = self.broker_maintenance['daily']
        if weekday in daily_maint['days']:
            if daily_maint['start'] <= current_time <= daily_maint['end']:
                end_time = check_time.replace(
                    hour=int(daily_maint['end'].split(':')[0]),
                    minute=int(daily_maint['end'].split(':')[1]),
                    second=0, microsecond=0
                )
                return {
                    'is_maintenance': True,
                    'reason': 'Broker daily maintenance window',
                    'end_time': end_time
                }
        
        # Check weekly maintenance (Sunday)
        if weekday == 6:  # Sunday
            weekly_maint = self.broker_maintenance['weekly']
            if weekly_maint['start'] <= current_time <= weekly_maint['end']:
                end_time = check_time.replace(
                    hour=int(weekly_maint['end'].split(':')[0]),
                    minute=int(weekly_maint['end'].split(':')[1]),
                    second=0, microsecond=0
                )
                return {
                    'is_maintenance': True,
                    'reason': 'Broker weekly maintenance window',
                    'end_time': end_time
                }
        
        return {'is_maintenance': False}

    def _get_next_market_open(self, current_time: datetime) -> datetime:
        """Get the next market opening time"""
        # If it's weekend, next open is Sunday 21:00 UTC
        weekday = current_time.weekday()
        
        if weekday == 5:  # Saturday
            # Next Sunday at 21:00
            days_to_add = 1
            next_open = current_time + dt.timedelta(days=days_to_add)
            return next_open.replace(hour=21, minute=0, second=0, microsecond=0)
        elif weekday == 6 and current_time.hour < 21:  # Sunday before 21:00
            return current_time.replace(hour=21, minute=0, second=0, microsecond=0)
        
        # Otherwise, market should be open or will open soon
        return current_time + dt.timedelta(hours=1)

    def _get_next_trading_day(self, current_time: datetime) -> datetime:
        """Get next trading day (skip weekends and holidays)"""
        next_day = current_time + dt.timedelta(days=1)
        
        # Skip weekends
        while next_day.weekday() >= 5:  # Saturday or Sunday
            next_day += dt.timedelta(days=1)
        
        return next_day.replace(hour=0, minute=0, second=0, microsecond=0)

    def _get_next_high_liquidity_time(self, base_currency: str, quote_currency: str, current_time: datetime) -> datetime:
        """Get next time when high liquidity is expected"""
        # For simplicity, return next major session opening
        
        # Check each upcoming hour for the next 24 hours
        for i in range(24):
            check_time = current_time + dt.timedelta(hours=i)
            active_sessions = self._get_active_sessions(check_time)
            liquidity = self._assess_liquidity(base_currency, quote_currency, active_sessions)
            
            if liquidity['level'] in ['high', 'medium']:
                return check_time.replace(minute=0, second=0, microsecond=0)
        
        # Fallback: next day at market open
        return self._get_next_market_open(current_time)

    def get_trading_recommendation(self, instrument: str, check_time: Optional[datetime] = None) -> Dict:
        """
        Get trading recommendation based on market availability and liquidity.
        
        Returns:
            Dict with recommendation, risk level, and suggested actions
        """
        market_status = self.is_market_open(instrument, check_time)
        
        recommendation = {
            'instrument': instrument,
            'recommendation': 'wait',
            'risk_level': 'high',
            'suggested_actions': [],
            'market_status': market_status
        }
        
        if not market_status['is_available']:
            recommendation['recommendation'] = 'wait'
            recommendation['risk_level'] = 'very_high'
            recommendation['suggested_actions'] = [
                f"Wait until market opens: {market_status.get('next_open', 'Unknown')}",
                f"Reason: {market_status['reason']}"
            ]
        else:
            liquidity_level = market_status['liquidity_level']
            
            if liquidity_level == 'high':
                recommendation['recommendation'] = 'trade'
                recommendation['risk_level'] = 'low'
                recommendation['suggested_actions'] = [
                    "Optimal trading conditions",
                    "Normal spreads expected",
                    "Good liquidity available"
                ]
            elif liquidity_level == 'medium':
                recommendation['recommendation'] = 'trade_with_caution'
                recommendation['risk_level'] = 'medium'
                recommendation['suggested_actions'] = [
                    "Acceptable trading conditions",
                    "Monitor spreads carefully",
                    "Consider smaller position sizes"
                ]
            elif liquidity_level == 'low':
                recommendation['recommendation'] = 'avoid_or_minimal'
                recommendation['risk_level'] = 'high'
                recommendation['suggested_actions'] = [
                    "Low liquidity conditions",
                    "Expect wider spreads",
                    "Use very small position sizes if trading",
                    "Consider waiting for better conditions"
                ]
            else:
                recommendation['recommendation'] = 'wait'
                recommendation['risk_level'] = 'very_high'
                recommendation['suggested_actions'] = [
                    "Very poor trading conditions",
                    "Wait for higher liquidity period"
                ]
        
        return recommendation

    def log_market_status(self, instrument: str, check_time: Optional[datetime] = None):
        """Log current market status for an instrument"""
        status = self.is_market_open(instrument, check_time)
        recommendation = self.get_trading_recommendation(instrument, check_time)
        
        log_msg = (
            f"Market Status for {instrument}: "
            f"Available={status['is_available']}, "
            f"Status={status['status']}, "
            f"Liquidity={status['liquidity_level']}, "
            f"Recommendation={recommendation['recommendation']}, "
            f"Risk={recommendation['risk_level']}"
        )
        
        if not status['is_available']:
            log_msg += f", Reason={status['reason']}"
            if status.get('next_open'):
                log_msg += f", NextOpen={status['next_open']}"
        
        self.logger.info(log_msg)
        return log_msg

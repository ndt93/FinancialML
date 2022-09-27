from enum import Enum, unique


class TickCol:
    TIMESTAMP = 'timestamp'
    PRICE = 'price'
    VOLUME = 'volume'


class BarCol:
    TIMESTAMP = 'timestamp'
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'
    CLOSE = 'close'
    VOLUME = 'volume'
    VWAP = 'vwap'


@unique
class BarUnit(Enum):
    TIME = 1
    TICK = 2
    VOLUME = 3
    DOLLARS = 4

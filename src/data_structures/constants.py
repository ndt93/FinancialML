from enum import Enum, unique


class TickCol:
    TIMESTAMP = 'Timestamp'
    PRICE = 'Price'
    VOLUME = 'Volume'


class BarCol:
    TIMESTAMP = 'Timestamp'
    OPEN = 'Open'
    HIGH = 'High'
    LOW = 'Low'
    CLOSE = 'Close'
    VOLUME = 'Volume'
    VWAP = 'VWAP'


@unique
class BarUnit(Enum):
    TIME = 1
    TICK = 2
    VOLUME = 3
    DOLLARS = 4

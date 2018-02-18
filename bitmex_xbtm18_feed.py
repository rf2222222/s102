from bitmex_feed import *

# Basic use of websocket.
def run():
    logger = setup_logger()
    Feed.init()
    Instrument.init()
    Prediction.init()
    Roi.init()
    

    sym="XBTM18"
    inst=getInstrument(sym, sym, 'Blockchain', 'Blockchain', sym)
    block=BlockchainData([inst])
    # Instantiating the WS will make it connect. Be sure to add your api_key/api_secret.
    ws = BitMEXWebsocket(endpoint="wss://www.bitmex.com/realtime", symbol=sym,
                         api_key=None, api_secret=None, data=block)

    logger.info("Instrument data: %s" % ws.get_instrument())

    # Run forever
    while(ws.ws.sock.connected):
        logger.info("Ticker: %s" % ws.get_ticker())
        if ws.config['api_key']:
            logger.info("Funds: %s" % ws.funds())
        logger.info("Market Depth: %s" % ws.market_depth())
        logger.info("Recent Trades: %s\n\n" % ws.recent_trades())
        sleep(10)



if __name__ == "__main__":
    run()
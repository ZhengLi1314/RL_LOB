from threading import Thread, Lock
import pandas as pd
import numpy as np
import time
import os
from gym import Env
from gym.spaces import Discrete, Box

class CirList:
    def __init__(self, length):
        self.size = length
        self._table = [None]*length
        self.idx = 0
        self._counter = 0

    def insertData(self, data):
        self._counter += 1
        self._table[self.idx] = data
        self.idx = (self.idx+1) % self.size

    def getData(self):
        tail = self._table[0:self.idx]
        head = self._table[self.idx:]
        ret = head+tail
        return ret.copy()

    def isFull(self):
        return self._counter >= self.size

    def __repr__(self):
        return str(self.getData())


class OrderBook:
    def __init__(self, AskOrder, BidOrder, last_price):
        self.last_price = last_price
        idx = 0
        tmp = pd.DataFrame(columns=['price', 'size', 'type', 'last_price'])
        ls = AskOrder

        for order in ls[::-1]:
            tmp.loc[idx] = [order.price, order.size, 'Ask', last_price]
            idx += 1

        self.n_ask = idx
        ls = BidOrder

        for order in ls:
            tmp.loc[idx] = [order.price, order.size, 'Bid', last_price]
            idx += 1

        self.n_bid = idx - self.n_ask

        self.df = tmp

    def __repr__(self):
        return str(self.df)

class SHIFT_env:
    def __init__(self,
                 trader,
                 t,
                 nTimeStep,
                 ODBK_range,
                 symbol,
                 target_price,
                 commission = 0,
                 rebate = 0,
                 save_data = True):
        
        #newly added##############
        #objective: control the mid price
        self.target_mid_price = target_price
        #bp threshold:
        self.bp_thres = 50000

        self.timeInterval = t
        self.symbol = symbol
        self.nTimeStep = nTimeStep
        self.ODBK_range = ODBK_range
        self.trader = trader
        self.commission = commission
        self.rebate = rebate
        self.mutex = Lock()
        

        self.dataThread = Thread(target=self._link)
        # features = ['symbol', 'orderPrice', 'orderSize', 'OrderTime']
        self.table = CirList(nTimeStep) #contains: 'curr_mp', 'volume_ask', 'volume_bid', 'remained_bp'
        self._cols = ['reward', 'order_type', 'price', 'size','curr_mp', 'volume_ask', 
                        'volume_bid', 'remained_bp', 'past_mean_mp', 'mp_vol']
        #['BA_spead', 'last_traded_price', 'Smart_price', 'Liquidity_imb', 'market_cost',
        #        'remained_shares', 'remained_time', 'reward', 'order_type','is_buy', 'premium',
        #        'obj_price', 'base_price', 'executed', 'done']
        self.df = pd.DataFrame(columns=self._cols)
        self.df_idx = 0

        print('Waiting for connection', end='')
        for _ in range(5):
            time.sleep(1)
            print('.', end='')
        print()

        self.thread_alive = True
        self.dataThread.start()

        self.remained_share = None
        self.total_share = None
        self.currentPos = None
        self.objPos = None
        self.isBuy = None
        self.remained_time = None
        self.tmp_obs = [None]*7
        self.name = 'exec_one_asset'
        self.isSave = save_data

        #new:
        self.current_state_info = []

    def set_objective(self, share, remained_time, premium = None):
        self.remained_share = abs(share)
        self.total_share = self.remained_share
        self.currentPos = self._getCurrentPosition()
        self.objPos = self.currentPos + share
        self.remained_time = remained_time
        self.isBuy = True if share> 0 else False
        self.premium = premium if premium else remained_time / 100

    @staticmethod
    def action_space():
        return 3

    @staticmethod
    def state_space():
        return 4

    #thread constantly collecting order book data and Last Price
    def _link(self):
        while self.trader.isConnected() and self.thread_alive:

            #last_price = self.trader.getLastPrice(self.symbol)

            Ask_ls = self.trader.getOrderBook(self.symbol, shift.OrderBookType.GLOBAL_ASK, self.ODBK_range)
            # assert Ask_ls, f'getOrderBook: return empty list: {self.symbol}-ASK-{self.ODBK_range}'

            Bid_ls = self.trader.getOrderBook(self.symbol, shift.OrderBookType.GLOBAL_BID, self.ODBK_range)
            # assert Bid_ls, f'getOrderBook: return empty list: {self.symbol}-BID-{self.ODBK_range}'
            
            #get remaining buying power
            bp = self.trader.get_portfolio_summary().get_total_bp()

            #get best bid and ask prices
            best_p = trader.get_best_price("CS2")

            info = self.LOB_to_list(Ask_ls, Bid_ls, best_p, bp)

            self.mutex.acquire()
            # print(88)
            self.table.insertData(info)
            # print(tmp)
            self.mutex.release()

            time.sleep(self.timeInterval)
        print('Data Thread stopped.')
    
    def LOB_to_list(self, ask, bid, best_p, bp): #return a list with these info 'curr_mp', 'volume_ask', 'volume_bid', 'remained_bp'
        #spread
        #sp = round((best_p.get_ask_price() - best_p.get_bid_price()),3)
        
        #mid price
        mid = round(((best_p.get_ask_price()+best_p.get_bid_price())/2),3)

        bid_size = 0
        ask_size = 0
        for order in ask_book:
            ask_size += order.size
        for order in bid_book:
            bid_size += order.size
        return list(mid, ask_size, bid_size, bp)

    def compute_state_info(self):
        #return the following items 'curr_mp', 'volume_ask', 'volume_bid', 'remained_bp', 'past_mean_mp', 'mp_vol'
        tab = self.table
        if tab.isFull():
            his_mp = []
            for ele in tab.getData():
                his_mp.append(ele[0])
            his_mp_np = np.array(his_mp)
            past_mean_mp = np.mean(his_mp_np)
            mp_vol = np.std(his_mp_np)
            return (tab.getData()[self.nTimeStep-1] + [past_mean_mp, mp_vol])
        else:
            print("need to wait for table to fill up")
        
    def step(self, order_type, price, size): #None if no action
        #apply action:##############################################################################################################################################
        #order_type: 1(Mar Sell) 2(Mar Buy) 3(Lmt Sell) 4(Lmt Buy)
        if order_type != None:
            if order_type <= 2:
                order = shift.Order(order_type, self.symbol, size)
                self.trader.submitOrder(order)
            else:
                order = shift.Order(order_type, self.symbol, size, price)
                self.trader.submitOrder(order)
        #wait for the order to have effect on LOB
        sleep(self.timeInterval)

        #cancell orders if the orders are not on the best levels: ??????????????

        #STATES: 4 components: ##################################################################################################################
        #    target mid price and real mid price diff | price stability | ask-bid balance | is bp at risk 
        state_info = self.compute_state_info() #: 'curr_mp', 'volume_ask', 'volume_bid', 'remained_bp', 'past_mean_mp', 'mp_vol'
        self.current_state_info = state_info
        state = self.get_states()

        #calculate reward: ###########################################################################################################################################
        reward =     0.75 * max(0, (10 - state[0]))    +     0.25 * max(0, 10 * (1 - abs(0.4 - state[1])))    -    state[3] * 10

        done = False

        #save
        if self.isSave:
            #['reward', 'order_type', 'price', 'size','curr_mp', 'volume_ask', 'volume_bid', 'remained_bp', 'past_mean_mp', 'mp_vol']
            tmp = [reward, order_type, price, size] + state_info

            self.df.loc[self.df_idx] = tmp
            self.df_idx += 1

        return state, reward, done, dict()

    def get_states(self):
        #target mid price and real mid price diff
        mp_diff = 0.6 * (self.current_state_info[0] - self.target_mid_price) + 0.4 * (self.current_state_info[4] - self.target_mid_price)

        #price stability
        mp_vol = self.current_state_info[5]

        #ask-bid balance:
        ab_bal = self.current_state_info[1] - self.current_state_info[2]

        #is bp at risk
        is_bp_risk = 0
        if self.current_state_info[3] <= self.bp_thres:
            is_bp_risk = 1
        
        return np.array(mp_diff, mp_vol, ab_bal, is_bp_risk)

    def step_afsadf(self):
        #premium = self.premium
        #print(f'premium: {premium}')
        #signBuy = 1 if self.isBuy else -1
        #base_price = self._getClosePrice(self.remained_share)
        #obj_mid_price = base_price - signBuy * premium

        print(f'base price: {base_price}, obj price: {obj_price}')

        if self.remained_time > 0:
            orderType = shift.Order.LIMIT_BUY if self.isBuy else shift.Order.LIMIT_SELL
        else:
            orderType = shift.Order.MARKET_BUY if self.isBuy else shift.Order.MARKET_SELL

        order = shift.Order(orderType, self.symbol, self.remained_share, obj_price)
        self.trader.submitOrder(order)
        print(f'submited: {order.symbol}, {order.type}, {order.price}, {order.size}, {order.id}, {order.timestamp}')
        time.sleep(self.timeInterval)

        print(f'waiting list size : {len(self.trader.getWaitingList())}')
        if self.trader.getWaitingListSize() > 0:
            self._cancelAllOrder(order)

        tmp_share = self.remained_share
        self.remained_share = self.total_share - abs(self._getCurrentPosition() - self.currentPos)
        exec_share = tmp_share - self.remained_share
        print(f'remain: {self.remained_share}, executed: {exec_share}, current: {self._getCurrentPosition()}')
        done = False
        if self.remained_time > 0:
            if premium > 0:
                reward = exec_share * premium * 100 + exec_share * self.rebate * 100
            else:
                reward = exec_share * premium * 100 - exec_share * self.commission * 100
        else:
            reward = exec_share * 0 - exec_share * 0.3
            done = True

        if self._getCurrentPosition() - self.objPos == 0:
            done = True
        self.remained_time -= 1
        self.premium -= 0.01

        if self.isSave:
            tmp = self.tmp_obs.tolist() + [reward, orderType, self.isBuy,
                                           premium, obj_price, base_price, exec_share, done]
            # print('-------------', self.tmp_obs)
            self.df.loc[self.df_idx] = tmp
            self.df_idx += 1

        next_obs = self._get_obs()
        return next_obs, reward, done, dict()

    def _get_orderID(self):
        ID = []
        for i in self.trader.getSubmittedOrders():
            ID.append(i.id)
        return ID[-1]

    def _cancelAllOrder(self, order):
        if order.type == shift.Order.LIMIT_BUY or order.type == shift.Order.MARKET_BUY:
            order.type = shift.Order.CANCEL_BID
        elif order.type == shift.Order.LIMIT_SELL or order.type == shift.Order.MARKET_SELL:
            order.type = shift.Order.CANCEL_ASK
        else:
            raise TypeError

        tmp_con = 0
        self.trader.submitOrder(order)
        print('Canceling order:', end='')
        #order_id = self._get_orderID()
        #status = self.trader.getOrder(order_id).status
        # while status != 'Status.CANCELLED':
        #     tmp_con += 1
        #     time.seelp(0.05)
        #     if tmp_con > 1000:
        #         print(f'\n current order info: %6s\t%21s\t%7.2f\t\t%4d\t%36s\t%26s' %
        #               (order.symbol, order.type, order.price, order.size, order.id, order.timestamp))
        #         print("Symbol\t\t\t\t\t Type\t  Price\t\tSize\tID\t\t\t\t\t\t\t\t\t\tTimestamp")
        #         for od in self.trader.getWaitingList():
        #             print("%6s\t%21s\t%7.2f\t\t%4d\t%36s\t%26s" %
        #                   (od.symbol, od.type, od.price, od.size, od.id, od.timestamp))
        #         raise TimeoutError(f'Waited for canceling order for {tmp_con * 0.05} seconds.')
        while self.trader.getWaitingListSize() > 0:
            tmp_con += 1
            time.sleep(0.05)
            print(self.trader.getWaitingListSize(), end='')
            if tmp_con > 1000:
                print(f'\n current order info: %6s\t%21s\t%7.2f\t\t%4d\t%36s\t%26s' %
                      (order.symbol, order.type, order.price, order.size, order.id, order.timestamp))
                print("Symbol\t\t\t\t\t Type\t  Price\t\tSize\tID\t\t\t\t\t\t\t\t\t\tTimestamp")
                for od in self.trader.getWaitingList():
                    print("%6s\t%21s\t%7.2f\t\t%4d\t%36s\t%26s" %
                          (od.symbol, od.type, od.price, od.size, od.id, od.timestamp))
                raise TimeoutError(f'Waited for canceling order for {tmp_con * 0.05} seconds.')
        # while status == 'Status.CANCELLED' and self.trader.getWaitingListSize() == 0:
            print(' done.')

    def _get_obs(self):
        self.tmp_obs = np.concatenate((self.compute(), np.array([self.remained_share, self.remained_time])))
        return self.tmp_obs.copy()

    def _getClosePrice(self, share):
        return self.trader.getClosePrice(self.symbol, self.isBuy, abs(share))

    def reset(self):
        print(f'Holding shares: {self.trader.getPortfolioItem(self.symbol).getShares()}')
        print(f'Buying Power: {self.trader.getPortfolioSummary().getTotalBP()}')
        self.close_all()
        return self.get_states()

    def save_to_csv(self, epoch):
        try:
            self.df.to_csv(f'./iteration_info/itr_{epoch}.csv')
            self.df = pd.DataFrame(columns=self._cols)
        except FileNotFoundError:
            os.makedirs(f'./iteration_info/', exist_ok= True)
            self.df.to_csv(f'./iteration_info/itr_{epoch}.csv')
            self.df = pd.DataFrame(columns=self._cols)


    def kill_thread(self):
        self.thread_alive = False

 
    
    @staticmethod
    def _ba_spread(df, n_ask):
        spread = df.price[n_ask - 1] - df.price[n_ask]
        return spread

    @staticmethod
    def _price(df):
        return df.last_price[0]/1000

    @staticmethod
    def _smart_price(df, n_ask):
        price = (df['size'][n_ask] * df.price[n_ask - 1] + df['size'][n_ask - 1] * df.price[n_ask]) \
                / (df['size'][n_ask] + df['size'][n_ask - 1])
        return price/1000

    @staticmethod
    def _liquid_imbal(df, n_ask, n_bid, act_direction):
        if n_ask > n_bid:
            imbal = df['size'][n_ask:].sum() - df['size'][(n_ask - n_bid):n_ask].sum()
        else:
            imbal = df['size'][n_ask:(2 * n_ask)].sum() - df['size'][0:n_ask].sum()
        if act_direction == 'Sell':
            imbal = -imbal
        return imbal/1000

    @staticmethod
    def _market_cost(df, n_ask, n_bid, act_direction, shares, commission):
        if act_direction == 'Buy':
            counter = df['size'][n_ask-1]
            n_cross = 1
            while counter < shares and n_ask-1 >= n_cross:
                counter += df['size'][n_ask-1-n_cross]
                n_cross += 1
            if n_cross > 1:
                sub_size = np.array(df['size'][(n_ask-n_cross):n_ask])
                sub_price = np.array(df.price[(n_ask-n_cross):n_ask])
                sub_size[0] = shares - sum(sub_size) + sub_size[0]
                market_price = sub_size.dot(sub_price)/shares
                cost = shares*(market_price - df.price[n_ask] + df.price[n_ask-1]*commission)
            else:
                market_price = df.price[n_ask-1]
                cost = shares*(market_price*(1+commission)-df.price[n_ask])
        else:
            counter = df['size'][n_ask]
            n_cross = 1
            while counter < shares and n_cross <= n_bid-1:
                counter += df['size'][n_ask+n_cross]
                n_cross += 1
            if n_cross > 1:
                sub_size = np.array(df['size'][n_ask:(n_ask+n_cross)])
                sub_price = np.array(df.price[n_ask:(n_ask+n_cross)])
                sub_size[-1] = shares - sum(sub_size) + sub_size[-1]
                market_price = sub_size.dot(sub_price)/shares
                cost = shares*(market_price - df.price[n_ask-1] + df.price[n_ask]*commission)
            else:
                market_price = df.price[n_ask]
                cost = shares*(market_price*(1+commission) - df.price[n_ask-1])
        return cost/1000, market_price

    def close_all(self):
        share = self.trader.getPortfolioItem(self.symbol).getShares()
        BP = self.trader.getPortfolioSummary().getTotalBP()
        waitingStep = 0
        small_order = 1
        while share != 0:
            position = int(share / 100)
            orderType = shift.Order.MARKET_BUY if position < 0 else shift.Order.MARKET_SELL

            if share < 0 and BP < abs(share) * self.trader.getClosePrice(self.symbol, True, abs(position)):
                order = shift.Order(orderType, self.symbol, small_order)
                self.trader.submitOrder(order)
                small_order *= 2
            else:
                order = shift.Order(orderType, self.symbol, abs(position))
                self.trader.submitOrder(order)

            time.sleep(0.5)
            #print(trader.getPortfolioItem(symbol).getShares())
            #print(trader.getPortfolioSummary().getTotalBP())
            share = self.trader.getPortfolioItem(self.symbol).getShares()
            waitingStep += 1
            assert  waitingStep < 40


    def _getCurrentPosition(self):
        return int(self.trader.getPortfolioItem(self.symbol).getShares() / 100)

    def __del__(self):
        self.kill_thread()
"""

if __name__=='__main__':
    table = CirList(3)
    #self._cols = ['reward', 'order_type', 'price']
    table.insertData([1,2,3])
    table.insertData([1,4,3])
    table.insertData([1,6,3])
    table.insertData([1,6,1])
    print(table.getData())
    print(table._table)
    print(table._counter)
    print(table.isFull())

"""

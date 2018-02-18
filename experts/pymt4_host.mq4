/* (C) Copyright 2013 Rob Watson rmawatson [at] hotmail.com  and others.
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser General Public License
 * (LGPL) version 2.1 which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/lgpl-2.1.html
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * Contributors:
 *     Rob Watson ( rmawatson [at] hotmail )
 *
 */

#import "pymt4.dll"

bool pymt4_initialize(string windowName,int windowHandle);
bool pymt4_uninitialize(string windowName,int windowHandle);
bool pymt4_isinitialized();
bool pymt4_notify(string windowName,int windowHandle);
bool pymt4_notifyOnTick(string symbol,double bid,double ask);
int  pymt4_requestPendingCommand();

string pymt4_getStringArgument(string);
double pymt4_getDoubleArgument();
int    pymt4_getIntArgument();
bool   pymt4_getBoolArgument();

bool pymt4_setDoubleResult(double result,int error);
bool pymt4_setStringResult(string result,int error);
bool pymt4_setIntResult(int result,int error);
bool pymt4_setBoolResult(int result,int error );
//+------------------------------------------------------------------+
//| script program start function                                    |
//+------------------------------------------------------------------+
#define StrBuffer "                                                                                                                               "
#define CmdId_CheckShutdownCondition 100

#define CmdId_AccountBalance 1000
#define CmdId_AccountCredit 1001
#define CmdId_AccountCompany 1002
#define CmdId_AccountCurrency 1003
#define CmdId_AccountEquity 1004
#define CmdId_AccountFreeMargin 1005
#define CmdId_AccountFreeMarginCheck 1006
#define CmdId_AccountFreeMarginMode 1007
#define CmdId_AccountLeverage 1008
#define CmdId_AccountMargin 1009
#define CmdId_AccountName 1010
#define CmdId_AccountNumber 1011
#define CmdId_AccountProfit 1012
#define CmdId_AccountServer 1013
#define CmdId_AccountStopoutLevel 1014
#define CmdId_AccountStopoutMode 1015

#define CmdId_iBars 2000
#define CmdId_iBarShift 2001
#define CmdId_iClose 2002
#define CmdId_iHigh 2003
#define CmdId_iHighest 2004
#define CmdId_iLow 2005
#define CmdId_iLowest 2006
#define CmdId_iOpen 2007
#define CmdId_iTime 2008
#define CmdId_iVolume 2009

#define CmdId_OrderClose 3000
#define CmdId_OrderCloseBy 3001
#define CmdId_OrderClosePrice 3002
#define CmdId_OrderCloseTime 3003
#define CmdId_OrderComment 3004
#define CmdId_OrderCommission 3005
#define CmdId_OrderDelete 3006
#define CmdId_OrderExpiration 3007
#define CmdId_OrderLots 3008
#define CmdId_OrderMagicNumber 3009
#define CmdId_OrderModify 3010
#define CmdId_OrderOpenPrice 3011
#define CmdId_OrderOpenTime 3012
#define CmdId_OrderPrint 3013
#define CmdId_OrderProfit 3014
#define CmdId_OrderSelect 3015
#define CmdId_OrderSend 3016
#define CmdId_OrdersHistoryTotal 3017
#define CmdId_OrderStopLoss 3018
#define CmdId_OrdersTotal 3019
#define CmdId_OrderSwap 3020
#define CmdId_OrderSymbol 3021
#define CmdId_OrderTakeProfit 3022
#define CmdId_OrderTicket 3023
#define CmdId_OrderType 3024


#define CmdId_IsConnected 4000
#define CmdId_IsDemo 4001
#define CmdId_IsDllsAllowed 4002
#define CmdId_IsExpertEnabled 4003
#define CmdId_IsLibrariesAllowed 4004
#define CmdId_IsOptimization 4005
#define CmdId_IsStopped 4006
#define CmdId_IsTesting 4007
#define CmdId_IsTradeAllowed 4008
#define CmdId_IsTradeContextBusy 4009
#define CmdId_IsVisualMode 4010
#define CmdId_UninitializeReason 4011

#define CmdId_TerminalCompany 5000
#define CmdId_TerminalName 5001
#define CmdId_TerminalPath 5002

#define CmdId_Alert 6000
#define CmdId_Comment 6001
#define CmdId_GetTickCount 6002
#define CmdId_MarketInfo 6003
#define CmdId_PlaySound 6004
#define CmdId_Print 6005
#define CmdId_SendFTP 6006
#define CmdId_SendMail 6007
#define CmdId_SendNotification 6008

#define CmdId_TimeCurrent 7000
#define CmdId_TimeLocal 7001
#define CmdId_TimeGMT 7002
#define CmdId_TimeGMTOffset 7003

#define CmdId_RefreshRates 9000		

void checkSelectedTicket(int ticketId)
{
   static int lastTicketId = 0;
   
   if (ticketId != lastTicketId)
      OrderSelect(ticketId,SELECT_BY_TICKET);
      
   lastTicketId = ticketId;
}

bool evaluateCommand(int pendingCommandId)
{

   switch(pendingCommandId)
   {
   
	   /* Account Functions */   
	   
      case CmdId_AccountBalance:
         pymt4_setDoubleResult(AccountBalance(),GetLastError());
         break;
      case CmdId_AccountCredit:
         pymt4_setDoubleResult(AccountCredit(),GetLastError());
	      break;
      case CmdId_AccountCompany:
         pymt4_setStringResult(AccountCompany(),GetLastError());      
	      break;
      case CmdId_AccountCurrency:
         pymt4_setStringResult(AccountCurrency(),GetLastError());
	      break;
      case CmdId_AccountEquity:
         pymt4_setDoubleResult(AccountEquity(),GetLastError());
	      break;
      case CmdId_AccountFreeMargin:
         pymt4_setDoubleResult(AccountFreeMargin(),GetLastError());
	      break;
      case CmdId_AccountFreeMarginCheck:
         pymt4_setDoubleResult(AccountFreeMarginCheck(pymt4_getStringArgument(StrBuffer),pymt4_getIntArgument(),pymt4_getDoubleArgument()),GetLastError());      
         break;
      case CmdId_AccountFreeMarginMode:
         pymt4_setDoubleResult(AccountFreeMarginMode(),GetLastError());
	      break;
      case CmdId_AccountLeverage:
         pymt4_setIntResult(AccountLeverage(),GetLastError());
	      break;
      case CmdId_AccountMargin:
         pymt4_setDoubleResult(AccountMargin(),GetLastError());
	      break;
      case CmdId_AccountName:
         pymt4_setStringResult(AccountName(),GetLastError());
	      break;
      case CmdId_AccountNumber:
         pymt4_setIntResult(AccountNumber(),GetLastError());
	      break;
      case CmdId_AccountProfit:
         pymt4_setDoubleResult(AccountProfit(),GetLastError());
	      break;
      case CmdId_AccountServer:
         pymt4_setStringResult(AccountServer(),GetLastError());
	      break;
      case CmdId_AccountStopoutLevel:
         pymt4_setIntResult(AccountStopoutLevel(),GetLastError());
	      break;
      case CmdId_AccountStopoutMode:
         pymt4_setIntResult(AccountStopoutMode(),GetLastError());
	      break;
	      
	      
	   /* Trading Functions */
	      
      case CmdId_iBars:
         pymt4_setIntResult(iBars(pymt4_getStringArgument(StrBuffer),pymt4_getIntArgument()),GetLastError());
	      break;
      case CmdId_iBarShift:
         pymt4_setIntResult(iBarShift(pymt4_getStringArgument(StrBuffer),pymt4_getIntArgument(),pymt4_getBoolArgument()),GetLastError());
	      break;	      
      case CmdId_iClose:
         pymt4_setDoubleResult(iClose(pymt4_getStringArgument(StrBuffer),pymt4_getIntArgument(),pymt4_getIntArgument()),GetLastError());
	      break;	      
      case CmdId_iHigh:
         pymt4_setDoubleResult(iHigh(pymt4_getStringArgument(StrBuffer),pymt4_getIntArgument(),pymt4_getIntArgument()),GetLastError());
	      break;
      case CmdId_iHighest:
         pymt4_setDoubleResult(iHighest(pymt4_getStringArgument(StrBuffer),pymt4_getIntArgument(),pymt4_getIntArgument(),pymt4_getIntArgument(),pymt4_getIntArgument()),GetLastError());
	      break;
      case CmdId_iLowest:
         pymt4_setDoubleResult(iLowest(pymt4_getStringArgument(StrBuffer),pymt4_getIntArgument(),pymt4_getIntArgument()),GetLastError());
	      break;
      case CmdId_iOpen:
         pymt4_setDoubleResult(iOpen(pymt4_getStringArgument(StrBuffer),pymt4_getIntArgument(),pymt4_getIntArgument()),GetLastError());
	      break;
      case CmdId_iTime:
         pymt4_setDoubleResult(iTime(pymt4_getStringArgument(StrBuffer),pymt4_getIntArgument(),pymt4_getIntArgument()),GetLastError());
	      break;
      case CmdId_iVolume:
         pymt4_setDoubleResult(iVolume(pymt4_getStringArgument(StrBuffer),pymt4_getIntArgument(),pymt4_getIntArgument()),GetLastError());
         break;    	      	      	      	      	      	      
    
    
    /* Order Functions */
    
      case CmdId_OrderClose:
         pymt4_setBoolResult(
                                    OrderClose(
                                             pymt4_getIntArgument(),
                                             pymt4_getDoubleArgument(),
                                             pymt4_getDoubleArgument(),
                                             pymt4_getIntArgument(),
                                             pymt4_getIntArgument()
                                             ),
                                             GetLastError()
                                    );
	      break;
      case CmdId_OrderCloseBy:
	      pymt4_setBoolResult(OrderCloseBy(pymt4_getIntArgument(),pymt4_getDoubleArgument(),pymt4_getIntArgument()),GetLastError());
	      break;
      case CmdId_OrderClosePrice:
         checkSelectedTicket(pymt4_getIntArgument());
	      pymt4_setDoubleResult(OrderClosePrice(),GetLastError());
	      break;
      case CmdId_OrderCloseTime:
         checkSelectedTicket(pymt4_getIntArgument());
	      pymt4_setIntResult(OrderCloseTime(),GetLastError());
	      break;
      case CmdId_OrderComment:
         checkSelectedTicket(pymt4_getIntArgument());
	      pymt4_setStringResult(OrderComment(),GetLastError());
	      break;
      case CmdId_OrderCommission:
         checkSelectedTicket(pymt4_getIntArgument());
	      pymt4_setDoubleResult(OrderCommission(),GetLastError());
	      break;
      case CmdId_OrderDelete:
	      pymt4_setBoolResult(OrderDelete(pymt4_getIntArgument(),pymt4_getIntArgument()),GetLastError());
	      break;
      case CmdId_OrderExpiration:
	      checkSelectedTicket(pymt4_getIntArgument()); 
	      pymt4_setIntResult(OrderExpiration(),GetLastError());
	      break;
      case CmdId_OrderLots:
	      checkSelectedTicket(pymt4_getIntArgument()); 
	      pymt4_setDoubleResult(OrderLots(),GetLastError());
	      break;
      case CmdId_OrderMagicNumber:
	      checkSelectedTicket(pymt4_getIntArgument()); 
	      pymt4_setIntResult(OrderMagicNumber(),GetLastError());
	      break;
      case CmdId_OrderModify:
	      pymt4_setBoolResult(
	                             OrderModify(
	                                pymt4_getIntArgument(),
	                                pymt4_getDoubleArgument(),
	                                pymt4_getDoubleArgument(),
	                                pymt4_getDoubleArgument(),
	                                pymt4_getIntArgument(),
	                                pymt4_getIntArgument()
	                             ),GetLastError()
	                          );
	      break;
      case CmdId_OrderOpenPrice:
	      checkSelectedTicket(pymt4_getIntArgument()); 
	      pymt4_setDoubleResult(OrderOpenPrice(),GetLastError());
	      break;
      case CmdId_OrderOpenTime:
	      checkSelectedTicket(pymt4_getIntArgument()); 
	      pymt4_setIntResult(OrderOpenTime(),GetLastError());
	      break;
      case CmdId_OrderPrint:
	      checkSelectedTicket(pymt4_getIntArgument()); 
	      OrderPrint();
	      pymt4_setBoolResult(True,GetLastError());
	      break;
      case CmdId_OrderProfit:
	      checkSelectedTicket(pymt4_getIntArgument()); 
	      pymt4_setDoubleResult(OrderProfit(),GetLastError());
	      break;
      case CmdId_OrderSelect:
         
         bool result = OrderSelect(pymt4_getIntArgument(),pymt4_getIntArgument(),pymt4_getIntArgument());
         
         if(result) pymt4_setIntResult(OrderTicket(),GetLastError());
         else pymt4_setIntResult(-1,GetLastError());

	      break;
      case CmdId_OrderSend:
	      pymt4_setIntResult(
	                             OrderSend(
	                                pymt4_getStringArgument(StrBuffer),
	                                pymt4_getIntArgument(),
	                                pymt4_getDoubleArgument(),
	                                pymt4_getDoubleArgument(),
	                                pymt4_getIntArgument(),
	                                pymt4_getDoubleArgument(),
	                                pymt4_getDoubleArgument(),
	                                pymt4_getStringArgument(StrBuffer),
	                                pymt4_getIntArgument(),
	                                pymt4_getIntArgument(),
	                                pymt4_getIntArgument()
	                             ),GetLastError()
	                          );
	      break;
      case CmdId_OrdersHistoryTotal:
	      pymt4_setIntResult(OrdersHistoryTotal(),GetLastError());
	      break;
      case CmdId_OrderStopLoss:
         checkSelectedTicket(pymt4_getIntArgument());
	      pymt4_setDoubleResult(OrderStopLoss(),GetLastError());
	      break;
      case CmdId_OrdersTotal:
	      pymt4_setIntResult(OrdersTotal(),GetLastError());
	      break;
      case CmdId_OrderSwap:
         checkSelectedTicket(pymt4_getIntArgument());
	      pymt4_setDoubleResult(OrderStopLoss(),GetLastError());
	      break;
      case CmdId_OrderSymbol:
         checkSelectedTicket(pymt4_getIntArgument());
	      pymt4_setStringResult(OrderSymbol(),GetLastError());
	      break;
      case CmdId_OrderTakeProfit:
         checkSelectedTicket(pymt4_getIntArgument());
	      pymt4_setDoubleResult(OrderTakeProfit(),GetLastError());
	      break;
      case CmdId_OrderTicket:
         checkSelectedTicket(pymt4_getIntArgument());
	      pymt4_setIntResult(OrderTicket(),GetLastError());
	      break;
      case CmdId_OrderType:
         checkSelectedTicket(pymt4_getIntArgument());
	      pymt4_setIntResult(OrderType(),GetLastError());
	      break;    
    
      
      /* Checkup */
      case CmdId_IsConnected:
         bool res = IsConnected();
         Print(res);
	      pymt4_setBoolResult(IsConnected(),GetLastError());
	      break;
      case CmdId_IsDemo:
	      pymt4_setBoolResult(IsDemo(),GetLastError());
	      break;
      case CmdId_IsDllsAllowed:
	      pymt4_setBoolResult(IsDllsAllowed(),GetLastError());
	      break;
      case CmdId_IsExpertEnabled:
	      pymt4_setBoolResult(IsExpertEnabled(),GetLastError());
	      break;
      case CmdId_IsLibrariesAllowed:
	      pymt4_setBoolResult(IsLibrariesAllowed(),GetLastError());
	      break;
      case CmdId_IsOptimization:
	      pymt4_setBoolResult(IsOptimization(),GetLastError());
	      break;
      case CmdId_IsStopped:
	      pymt4_setBoolResult(IsStopped(),GetLastError());
	      break;
      case CmdId_IsTesting:
	      pymt4_setBoolResult(IsTesting(),GetLastError());
	      break;
      case CmdId_IsTradeAllowed:
	      pymt4_setBoolResult(IsTradeAllowed(),GetLastError());
	      break;
      case CmdId_IsTradeContextBusy:
	      pymt4_setBoolResult(IsTradeContextBusy(),GetLastError());
	      break;
      case CmdId_IsVisualMode:
	      pymt4_setBoolResult(IsVisualMode(),GetLastError());
	      break;
      case CmdId_UninitializeReason:
	      pymt4_setIntResult(UninitializeReason(),GetLastError());
	      break;
	      
	   /* Client Terminal */  
	      
      case CmdId_TerminalCompany:
	      pymt4_setStringResult(TerminalCompany(),GetLastError());
	      break;
      case CmdId_TerminalName:
	      pymt4_setStringResult(TerminalName(),GetLastError());
	      break;
      case CmdId_TerminalPath:
	      pymt4_setStringResult(TerminalPath(),GetLastError());
	      break;
      
      /* Common Functions */

      case CmdId_Alert:
	      Alert(pymt4_getStringArgument(StrBuffer));
	      pymt4_setBoolResult(True,GetLastError());
	      break;
      case CmdId_Comment:
	      Comment(pymt4_getStringArgument(StrBuffer));
	      pymt4_setBoolResult(True,GetLastError());
	      break;
      case CmdId_GetTickCount:
	      pymt4_setIntResult(GetTickCount(),GetLastError());
	      break;
      case CmdId_MarketInfo:
         Print("Start MarketInfo");
	      pymt4_setDoubleResult(MarketInfo(pymt4_getStringArgument(StrBuffer),pymt4_getIntArgument()),GetLastError());
	      Print("End MarketInfo");
	      break;
      case CmdId_PlaySound:
         PlaySound(pymt4_getStringArgument(StrBuffer));
	      pymt4_setBoolResult(True,GetLastError());
	      break;
      case CmdId_Print:
	      Print(pymt4_getStringArgument(StrBuffer));
	      pymt4_setBoolResult(True,GetLastError());
	      break;
      case CmdId_SendFTP:
	      pymt4_setBoolResult(SendFTP(pymt4_getStringArgument(StrBuffer),pymt4_getStringArgument(StrBuffer)),GetLastError());
	      break;
      case CmdId_SendMail:
	      SendMail(pymt4_getStringArgument(StrBuffer),pymt4_getStringArgument(StrBuffer));
	      pymt4_setBoolResult(True,GetLastError());
	      break;
      case CmdId_SendNotification:
	      pymt4_setBoolResult(SendNotification(pymt4_getStringArgument(StrBuffer)),GetLastError());  
	      break;
		  
 	   /* DateTime functions */
	   
	   case CmdId_TimeCurrent:	      
	      pymt4_setIntResult(TimeCurrent(),GetLastError());
	      break;
	   case CmdId_TimeLocal:	      
	      pymt4_setIntResult(TimeLocal(),GetLastError());
	      break;
	   case CmdId_TimeGMT:
	      pymt4_setIntResult(TimeGMT(),GetLastError());
	      break;
	   case CmdId_TimeGMTOffset:
	      pymt4_setIntResult(TimeGMTOffset(),GetLastError());
	      break;
		  
	   case CmdId_RefreshRates:	      
	      return pymt4_setBoolResult(RefreshRates(), GetLastError());
	      break;		
  }
   
    return(true);
}


int init()
{
   if(!pymt4_initialize(Symbol(),WindowHandle(Symbol(),Period())))
   {
      Print("PyMT4 Already Intialized ...");
      return(7);
   }  
   
   Print("PyMT4 Host Initialized ...");
   while(true)
   {
      int pendingCommandId = pymt4_requestPendingCommand();
      
      if (pendingCommandId != CmdId_CheckShutdownCondition)
         evaluateCommand(pendingCommandId);    

      
   
      if (IsStopped())
         break;   
   }

   pymt4_uninitialize(Symbol(),WindowHandle(Symbol(),Period()));
   Print("PyMT4 Host Uninitialized ...");

   return(False);
}


int deinit()
{ return(0); }

int start()
{ 
Print("Host Start");
return(0); 

}
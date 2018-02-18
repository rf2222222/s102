//+------------------------------------------------------------------+
//|                           Currency_Loader.mq4                    |
//|                      Copyright © 2006, Larionov P.V.             |
//|                                        lolion@mail.ru            |
//+------------------------------------------------------------------+
#property copyright "NSZ Foundation"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
//---- input parameters
//extern int  BarsMin=100;         // Minimal number bars in history which might be loaded into files.
//extern int MaxBarsInFile = 200000; // Max Bars for loading into file.
extern int FrequencyUpdate = 15; // this value identify frequency update for files in sec.
extern string signal;     //Signal File
extern double qty; // Order Quantity
extern bool AllowInfo = false;
extern bool AllowLogFile = false;

int pos=0;
string ExpertName = "Proc_Signal";

double ArrayM1[][6], ArrayM5[][6], ArrayM15[][6], ArrayM30[][6], ArrayH1[][6],ArrayH4[][6], ArrayD1[][6], ArrayW1[][6], ArrayMN[][6]; 
int ct, iDigits, Tryes=5, Pause=500, ArrSizeM1, ArrSizeM5, ArrSizeM15, ArrSizeM30, ArrSizeH1, ArrSizeH4, ArrSizeD1, ArrSizeW1, ArrSizeMN, i2, i3, h1, h2, h3, h4, h5, h6, h7, h8, h9, LCM1, LCM5, LCM15, LCM30, LCH1, LCH4, LCD1, LCW1, LCMN, LastError;
string CString, x, x2, FileSignal, FileBidAsk, FileBar, FileNameM1, FileNameM5, FileNameM15, FileNameM30, FileNameH1, FileNameH4, FileNameD1, FileNameW1, FileNameMN, FilePatch, FirstLine, BidAskFirstLine;

int OnInit()
  {
//--- create timer
   iDigits=MarketInfo(Symbol(),MODE_DIGITS);
   x="\"";
   x2="\\";
   FilePatch = ""; 
   FileSignal="signals\\" + signal + ".csv";
   printf( "Signal File:"+ FileSignal);
   ct=0;
   EventSetTimer(FrequencyUpdate);
      
   return(INIT_SUCCEEDED);
}

int read_signal() {
   string separator=",";
   int handle=-1, count, line_count = 1;
   string m_filename=FileSignal;
   //handle=FileOpen(m_filename, FILE_SHARE_READ|FILE_CSV|FILE_ANSI|FILE_READ,separator,CP_ACP);
   handle=FileOpen(m_filename, FILE_SHARE_READ|FILE_ANSI|FILE_READ);
   if(handle<0)
     {
     Print("I can't open the file.");
     }
     else
     {
        Info("File successfully open.");
        string allLines[5];
        
        //StringSplit(strFile,'\n',allLines);
        int line=0;
;
        while(!FileIsEnding(handle)) {
               int szFile   = (int)FileSize(handle);
               string strFile  = FileReadString(handle, szFile);
               allLines[line]=strFile;
               if (FileIsLineEnding(handle))
                  line++;
                  if ( line >= ArraySize(allLines)) {
                     ArrayResize(allLines, line+1);
                  }
        
        }
        FileClose(handle);
        int LineNum=ArraySize(allLines);
        Info("Found " + IntegerToString(LineNum) + " Lines\n");
        string cols[];
        StringSplit(allLines[0],',', cols);
        Info( allLines[0] + "\n\n");
        Info(allLines[LineNum - 2]);
        int signalcol=0;
        int safefcol=0;
        string sigcolname="signals";
        string safefcolname="safef";
        string signals[];
        StringSplit(allLines[LineNum-2],',',signals);
        for (int i=0; i < ArraySize(signals); i++) {
            string val=StringTrimRight(StringTrimLeft(cols[i]));
            if (StringCompare(sigcolname, val) == 0) {
               signalcol=i;   
               Info("Found Signal Col: " + IntegerToString(signalcol));
            }
            if (StringCompare(safefcolname, val) == 0) {
               safefcol=i;  
               Info("Found Safef Col: " + IntegerToString(safefcol)); 
            }
            Info("Col: " + val + " Val: " + (signals[i]) + "\n");
        }
       
        pos=StrToDouble(signals[signalcol])*StrToDouble(signals[safefcol]);
        Info("Found Signa Pos: " + DoubleToString(pos));
        
       
     }
     return pos;
        
}

int get_mt4_pos() {
   int mt4pos=0;
   for (int ii=OrdersTotal()-1 ; ii>=0 ; ii--)
   {
      if (!OrderSelect(ii,SELECT_BY_POS)) continue;
      if (OrderSymbol() == Symbol()) {
              
               int order_type=OrderType();
               
               if (order_type == OP_SELL || 
                   order_type == OP_SELLLIMIT || 
                   order_type == OP_SELLSTOP) {
                  mt4pos-=OrderLots();
               }else {
                  mt4pos+=OrderLots();
              
               }
        }
               
    }
    Info("Found Mt4 Pos: " + IntegerToString(mt4pos));
    return mt4pos;
   
}

void place_order(string action, double quant) {
   
    for (int i=0 ; i < OrdersTotal() ; i++)
   {
      if (!OrderSelect(i,SELECT_BY_POS)) 
         continue;
      if (OrderSymbol() == Symbol()) {
            if (StringCompare("STC",action)==0 || 
                StringCompare("BTC",action)==0) {
                 if (quant > 0) {
                    int order_type=OrderType();
                    double closeqty=0;
                    if (quant >= OrderLots())
                     closeqty=OrderLots();
                    else
                     closeqty = quant;
                    double price=Ask;
                    if (order_type == OP_SELL || 
                         order_type == OP_SELLLIMIT || 
                         order_type == OP_SELLSTOP) {
                        price=Ask;
                       
                     } else {
                        price=Bid;
                    
                     }
                     quant-=closeqty;
                     int slippage=3;
                     color arrow_color=0xFFEEFF;
                     
                     int ticket=  OrderTicket();
                     OrderClose(
                                   ticket,      // ticket
                                   closeqty,        // volume
                                   price,       // close price
                                   slippage,    // slippage
                                   arrow_color  // color
                        );
                     //double profit=OrderTakeProfit();
                    
                  }
                
            }
         }
    }
   if (StringCompare("BTO",action)==0) {
             double orderqty=quant;
             Info("Placing BUY Order for: " + DoubleToStr(orderqty));
             OrderSend(Symbol(),OP_BUY,orderqty,Ask,3,0, 999999);
   } else if (StringCompare("STO",action)==0) {
             double orderqty=quant;
             Info("Placing SELL Order for: " + DoubleToStr(orderqty));
             OrderSend(Symbol(),OP_SELL,orderqty,Bid,3,999999,0);
   }
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- destroy timer
   EventKillTimer();
      
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
//---
    
    
}

bool ROpen=false;
void OnTimer()
  {
      ROpen=true;
      Info("OnTimer\n");
      double syspos=read_signal() * qty;
      double mt4pos=get_mt4_pos();
      
      if (syspos > mt4pos) {
         double orderquant=syspos - mt4pos;
         if (mt4pos < 0){        
             double qty=MathMin(MathAbs(mt4pos), MathAbs(mt4pos - syspos));
             Info( "BTC " + DoubleToStr(qty) );
             place_order("BTC", qty);
             orderquant = orderquant - qty;
          }
           
         if (orderquant > 0){
             Info( "BTO"+ DoubleToStr(orderquant) );
             place_order("BTO", orderquant);
          }
      }
      if (syspos < mt4pos){
         double orderquant=mt4pos - syspos;
         
         if (mt4pos > 0) {        
             qty=MathMin(MathAbs(mt4pos), MathAbs(mt4pos - syspos));
             Info( "STC" + DoubleToStr(qty) );
             place_order("STC", qty);
             orderquant = orderquant - qty;
          }

         if (orderquant > 0){
             Info( "STO" + DoubleToStr(orderquant) );
             place_order("STO", orderquant);
          }
       }
      ROpen=false;
      
  }
//+------------------------------------------------------------------+
//| Tester function                                                  |
//+------------------------------------------------------------------+
double OnTester()
 {
   double ret=0.0;
   printf("OnTester\n");
   return(ret);
}

double Info( string MessageBody="HI"){//1
int WarnMessLevel=3;
string MessageType="1,0,0";
 int Actuality = 1;
  string MessageHead="";
bool AllowMailSending=false, AllowStatement=false;
string _CurTimeDaily;
 string CurRusTimeMin;
 _CurTimeDaily = GetCurRusTime("Days"); 
 if(Actuality<=WarnMessLevel){//2
 int i, fwr;
 int OpString=-1;
 string FileName =StringConcatenate("LOG",ExpertName,_CurTimeDaily); 
 if(MessageHead=="")MessageHead=StringConcatenate("from expert ",ExpertName,TimeToStr(CurTime()));
 OpString = StringFind(MessageType,"1",0);
 if(OpString>-1){if(AllowInfo){Print(MessageBody);}}
 OpString = StringFind(MessageType,"2",0);
  if(OpString>-1){//3
   if(AllowLogFile){//4
    CurRusTimeMin = GetCurRusTime("Seconds")+" ";
    MessageBody= StringConcatenate(CurRusTimeMin,MessageBody); 
    for(i=0; i<5; i++){//5
    int HFile=FileOpen(FileName,FILE_READ|FILE_WRITE," "); 
     if(HFile>0){//6    
     FileSeek(HFile,0,SEEK_END); 
     FileWrite(HFile,MessageBody); 
     FileFlush(HFile); 
     FileClose(HFile);
     break; 
     }else{Sleep(500); continue; }//6
    }//5
   }//4
  }//3
  OpString = StringFind(MessageType,"3",0);
  if(OpString>-1){//3
   if(AllowMailSending && (!IsTesting())){//4
   int RetVal;
   SendMail( MessageHead, MessageBody );
   RetVal = GetLastError();
   if(RetVal>0){Info("Error sending "+ErrorDescription(RetVal));}
   }//4
  }//3
  OpString = StringFind(MessageType,"4",0);
  if(OpString>-1){//3
   if(AllowStatement){//4
   i=i;
   }//4
  }//3
 }else{return(0);}//2
return(0);
}//1


string GetCurRusTime(string Detail) 
{//1 
   string StrMonth="",StrDay="",StrHour="",StrMinute="",StrSeconds=""; 
   RefreshRates(); 
 
 if (Detail == "Seconds"){  
   if(Month()<10) { StrMonth="0"+Month(); } else { StrMonth=Month(); } 
   if(Day()<10) { StrDay="0"+Day(); } else { StrDay=Day(); } 
   if(Hour()<10) { StrHour="0"+Hour(); } else { StrHour=Hour(); } 
   if(Minute()<10) { StrMinute="0"+Minute(); } else { StrMinute=Minute(); } 
   if(Seconds()<10) { StrSeconds="0"+Seconds(); } else { StrSeconds=Seconds(); } 
   return(""+StrDay+"."+StrMonth+"."+Year()+" "+StrHour+":"+StrMinute+":"+StrSeconds+" ");  
   }
 if (Detail == "Hours"){  
   if(Month()<10) { StrMonth="0"+Month(); } else { StrMonth=Month(); } 
   if(Day()<10) { StrDay="0"+Day(); } else { StrDay=Day(); } 
   if(Hour()<10) { StrHour="0"+Hour(); } else { StrHour=Hour(); } 
   if(Minute()<10) { StrMinute="0"+Minute(); } else { StrMinute=Minute(); } 
   if(Seconds()<10) { StrSeconds="0"+Seconds(); } else { StrSeconds=Seconds(); } 
   return(""+StrDay+"."+StrMonth+"."+Year()+" "+StrHour+":00:"+"00 ");  
   }
 if (Detail == "Days"){  
   if(Month()<10) { StrMonth="0"+Month(); }else { StrMonth=Month(); } 
   if(Day()<10) { StrDay="0"+Day(); } else { StrDay=Day(); } 
   if(Hour()<10) { StrHour="0"+Hour(); } else { StrHour=Hour(); } 
   if(Minute()<10) { StrMinute="0"+Minute(); } else { StrMinute=Minute(); } 
   if(Seconds()<10) { StrSeconds="0"+Seconds(); } else { StrSeconds=Seconds(); } 
   return(""+StrDay+"."+StrMonth+"."+Year()+" ");  
   }
   return Detail;
}//1 

//---- codes returned from trade server
string ErrorDescription(int error_code)
  {
   string error_string;
//----
   switch(error_code)
     {
      case 0:
      case 1:   error_string="no error";                                                  break;
      case 2:   error_string="common error";                                              break;
      case 3:   error_string="invalid trade parameters";                                  break;
      case 4:   error_string="trade server is busy";                                      break;
      case 5:   error_string="old version of the client terminal";                        break;
      case 6:   error_string="no connection with trade server";                           break;
      case 7:   error_string="not enough rights";                                         break;
      case 8:   error_string="too frequent requests";                                     break;
      case 9:   error_string="malfunctional trade operation";                             break;
      case 64:  error_string="account disabled";                                          break;
      case 65:  error_string="invalid account";                                           break;
      case 128: error_string="trade timeout";                                             break;
      case 129: error_string="invalid price";                                             break;
      case 130: error_string="invalid stops";                                             break;
      case 131: error_string="invalid trade volume";                                      break;
      case 132: error_string="market is closed";                                          break;
      case 133: error_string="trade is disabled";                                         break;
      case 134: error_string="not enough money";                                          break;
      case 135: error_string="price changed";                                             break;
      case 136: error_string="off quotes";                                                break;
      case 137: error_string="broker is busy";                                            break;
      case 138: error_string="requote";                                                   break;
      case 139: error_string="order is locked";                                           break;
      case 140: error_string="long positions only allowed";                               break;
      case 141: error_string="too many requests";                                         break;
      case 145: error_string="modification denied because order too close to market";     break;
      case 146: error_string="trade context is busy";                                     break;
      //---- mql4 errors
      case 4000: error_string="no error";                                                 break;
      case 4001: error_string="wrong function pointer";                                   break;
      case 4002: error_string="array index is out of range";                              break;
      case 4003: error_string="no memory for function call stack";                        break;
      case 4004: error_string="recursive stack overflow";                                 break;
      case 4005: error_string="not enough stack for parameter";                           break;
      case 4006: error_string="no memory for parameter string";                           break;
      case 4007: error_string="no memory for temp string";                                break;
      case 4008: error_string="not initialized string";                                   break;
      case 4009: error_string="not initialized string in array";                          break;
      case 4010: error_string="no memory for array\' string";                             break;
      case 4011: error_string="too long string";                                          break;
      case 4012: error_string="remainder from zero divide";                               break;
      case 4013: error_string="zero divide";                                              break;
      case 4014: error_string="unknown command";                                          break;
      case 4015: error_string="wrong jump (never generated error)";                       break;
      case 4016: error_string="not initialized array";                                    break;
      case 4017: error_string="dll calls are not allowed";                                break;
      case 4018: error_string="cannot load library";                                      break;
      case 4019: error_string="cannot call function";                                     break;
      case 4020: error_string="expert function calls are not allowed";                    break;
      case 4021: error_string="not enough memory for temp string returned from function"; break;
      case 4022: error_string="system is busy (never generated error)";                   break;
      case 4050: error_string="invalid function parameters count";                        break;
      case 4051: error_string="invalid function parameter value";                         break;
      case 4052: error_string="string function internal error";                           break;
      case 4053: error_string="some array error";                                         break;
      case 4054: error_string="incorrect series array using";                             break;
      case 4055: error_string="custom indicator error";                                   break;
      case 4056: error_string="arrays are incompatible";                                  break;
      case 4057: error_string="global variables processing error";                        break;
      case 4058: error_string="global variable not found";                                break;
      case 4059: error_string="function is not allowed in testing mode";                  break;
      case 4060: error_string="function is not confirmed";                                break;
      case 4061: error_string="send mail error";                                          break;
      case 4062: error_string="string parameter expected";                                break;
      case 4063: error_string="integer parameter expected";                               break;
      case 4064: error_string="double parameter expected";                                break;
      case 4065: error_string="array as parameter expected";                              break;
      case 4066: error_string="requested history data in update state";                   break;
      case 4099: error_string="end of file";                                              break;
      case 4100: error_string="some file error";                                          break;
      case 4101: error_string="wrong file name";                                          break;
      case 4102: error_string="too many opened files";                                    break;
      case 4103: error_string="cannot open file";                                         break;
      case 4104: error_string="incompatible access to a file";                            break;
      case 4105: error_string="no order selected";                                        break;
      case 4106: error_string="unknown symbol";                                           break;
      case 4107: error_string="invalid price parameter for trade function";               break;
      case 4108: error_string="invalid ticket";                                           break;
      case 4109: error_string="trade is not allowed";                                     break;
      case 4110: error_string="longs are not allowed";                                    break;
      case 4111: error_string="shorts are not allowed";                                   break;
      case 4200: error_string="object is already exist";                                  break;
      case 4201: error_string="unknown object property";                                  break;
      case 4202: error_string="object is not exist";                                      break;
      case 4203: error_string="unknown object type";                                      break;
      case 4204: error_string="no object name";                                           break;
      case 4205: error_string="object coordinates error";                                 break;
      case 4206: error_string="no specified subwindow";                                   break;
      default:   error_string="unknown error";
     }
//----
   return(error_string);
  }




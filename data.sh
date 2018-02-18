curl -XPUT 'localhost:9201/_all/_settings' -d '
{
"index.max_result_window" : "10000000"
}'
python bitmex_xbth18_feed.py &
python bitmex_xbtm18_feed.py &
python bitmex_xbtm18_data.py 
python bitmex_xbth18_data.py 

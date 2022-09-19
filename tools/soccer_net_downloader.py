import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="data/SoccerNET/downloads")

mySoccerNetDownloader.password = input("Password for videos (received after filling the NDA)")
# mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train","valid","test","challenge"])
# # mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train","valid","test","challenge"])


# mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train","test","challenge"])
# mySoccerNetDownloader.downloadGames(files=[f"1_HQ_25_player_bbox.json", f"2_HQ_25_player_bbox.json"])
mySoccerNetDownloader.downloadDataTask(task="calibration", split=["train","valid","test","challenge"])
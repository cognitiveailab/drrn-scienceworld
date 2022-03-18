#
# If ScienceWorld environments have been left open (from a crash, etc), 
# this will send close signals to any that exist across a large port range. 
#

from scienceworld import ScienceWorldEnv

MAX_THREADS = 200

for threadNum in range(0, MAX_THREADS):
    print("Thread num: " + str(threadNum) + " / " + str(MAX_THREADS))
    try:        
        env = ScienceWorldEnv("", None, 10, threadNum, launchServer=False)
        env.shutdown()
    except:
        print("\tNo Server Found")



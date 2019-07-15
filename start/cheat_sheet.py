import sys;
def readAndExit():
    while True:
        print("What do you want to do?");
        inString = raw_input()
        print(inString)
        if inString == "exit":
            sys.exit();
        else:
            print("Your input is {}".format(inString));


spam =["apple","banana","orange"];
print(spam[0]);
print(spam[-1]);

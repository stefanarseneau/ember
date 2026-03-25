Here is a zip file of the SED code. Could you run the second half of the white dwarfs? That would be 11,280 - 22,563.

To run the code, just type "python SEDs_WDMS_v5.py starting_num ending_num" in the directory where you store the file.

It should save the tables in the folder "tables" and the plots in the "plots" folder. It takes about 12 hours to run 300 so I have been running 4 codes at the same time each on 300 objects. Feel free to choose whatever you like. The code spits out a bunch of warnings about the length of the chain not being long enough but don't worry about those. I check if it is too short after a certain number of steps and then run more.
Let me know if it doesn't work and I can try to debug. I ran the first 1,200 last night with no problems on my machine.
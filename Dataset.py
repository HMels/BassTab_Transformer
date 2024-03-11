# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:19:44 2024

@author: Mels

This dataset will start with loading all the data that has been scraped off the internet. 
It will focus on splitting everything in simple vectors containing only one note. 
"""
#TODO maybe Look 2 beats forward to process special moves and digits bigger than 9
    # Add character for an empty note
    # this character should be skipped in outputs
#TODO right now a lot of errors. Missing data!

import requests
from bs4 import BeautifulSoup
import json

global number_dict
global special_dict
number_dict = [str(x) for x in range(0, 19)]
special_dict = ["h", "p", "s", "/", r'"\"', "x", ".","-","^","~","(n)","|", "G", "Gb","D","Db","A","Ab","E","Eb"]


def parse_tab_line(line):
    """
    Parses a single line of a tab and returns the notes until the third "|".
    If there are characters after the third "|" sign, it returns them separately as a repeat (the amount of times this bar is repeated).

    Parameters
    ----------
    line : str
        The tab line to be parsed.

    Returns
    -------
    output : str
        The notes in the line until the third "|".
    repeat : int or None
        The repeat count of the notes after the third "|" sign. If no repeat count is found, returns None.
    isempty : bool
        Is true when the complete bassline did not contain any notes
    """
    # Initialize variables
    output = ""
    pipe_count = 0
    remaining = ""
    isempty = True # will be used to check if there are digits in the bassline
    
    if line[-1]=="/": line = line[:-1] # some bug that's easier to delete like this
    if line[-1]!="|": line+="|" # make sure we end on a |
    
    # delete double ||
    result = ''
    for i in range(len(line)):
        if line[i] == '|' and i < len(line)-1 and line[i+1] == '|':
            continue  # skip the second '|' character
        result += line[i]
    line = result
    
    maxpipe_count = line.count("|")
    
    # Loop through characters in the line
    #TODO fix it for input that is different ("p1")
    try:
        line+="    " # to make sure indices exist
        for c in line:
            if c == "|":
                pipe_count += 1
                if pipe_count == maxpipe_count:
                    # Save remaining characters after third "|" sign
                    if maxpipe_count==2: remaining = line[line.index("|", line.index("|") + 1) + 1:]
                    else: remaining = line[line.index("|", line.index("|", line.index("|") + 1) + 1) + 1:]
                    output += c
                    break
            output += c
            if c.isdigit(): isempty = False
    except: remaining = ""
        
    # clean up line extra     
    remaining = remaining.replace(" ","")
    output = output.replace(" ","")
    
    '''
    # Check for repeat count in remaining characters
    repeat = 1
    try:
        if len(remaining)>0:
            remaining+="  "
            for r in remaining:
                if r == "x":
                    if remaining[remaining.index("x")+2].isdigit():
                        repeat = int(remaining[remaining.index("x")+1] + remaining[remaining.index("x")+2])
                    elif remaining[remaining.index("x")+1].isdigit():
                        repeat = int(remaining[remaining.index("x")+1])
                    else: repeat = None
                    break
    except: print('Minor error occured in the remainder')
    '''
    return output, 1, isempty


def load_bassTab(url):
    '''
    Loads a bass tab from a URL and returns an instance of the BassTab class containing the tab data.
    This function is as of right now very clunky and mainly focusses on getting all the tabs into similar 
    formats. 

    Parameters
    ----------
    url : str
        The URL of the bass tab to be loaded.
    
    Returns
    -------
    bassTab : BassTab
        An instance of the BassTab class containing the loaded tab data.
    '''
    bassTab = []
    
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        data_content = soup.find('div', {'class': 'js-store'})['data-content']
        data = json.loads(data_content)
    except: 
        print("URL cannot be loaded")
        return None
    
    try:
        if not data['store']['page']['data']['tab']['type']=='Bass Tabs':
            print('Warning, not a bass tab that is being loaded. Returning None')
            return None
    except: 
        pass
    
    try:
        if data['store']['page']['data']['tab_view']['meta']['tuning']['value']=='D A D G':
            print(data['store']['page']['data']['tab']["artist_name"],data['store']['page']['data']['tab']["song_name"],
                  "in drop-D tuning. So we skip it!")
            return None
    except:
        pass
    
    try:
        # Now you can access the data you are interested in using dictionary keys
        tab = data['store']['page']['data']['tab_view']['wiki_tab']['content']
    
        # Split the tablature text into individual lines
        tab_lines = tab.split('\n')
                
        # Extract the relevant information from the lines we're interested in
        Efound=False
        Afound=False
        Dfound=False
        Gfound=False

        for line0 in tab_lines:
            # We are filtering all useless information here, such as text about the tab and more
            # all the ways the program might fail
            line=""
            for char in line0:
                if char in number_dict or char in special_dict:
                    line += char
            line=line.replace('[/tab]',"")
            line=line.replace("[tab]","")
            line=line.replace(":","")
            line=line.replace(";","")
            line=line.replace("G","G|")
            line=line.replace("Gb","G|")
            line=line.replace("A","A|")
            line=line.replace("Ab","A|")
            line=line.replace("D","D|")
            line=line.replace("Db","D|")
            line=line.replace("E-mail","")
            line=line.replace("E","E|")
            line=line.replace("Eb","E|")
            line=line.replace("||","|")
            line=line.replace("||","|")
            if line[-2:]=="-\r":
                line=line.replace("-\r","-|\r")
            if line[-8:]=='-[/tab]\r':
                line=line.replace('-[/tab]\r','-|[/tab]\r')
            
            # cleaning up the data
            line=line.replace(" ","")
            line=line.replace('o',"-")
            line=line.replace('*',"-")
            line=line.replace('r',"-")
            line=line.replace("H","h")
            line=line.replace("P","p")
            line=line.replace("X","x")
            line=line.replace("S","s")
            
            # some tabs don't have tuning, so we need to add that 
            if line[:2]=='|-':
                if not Gfound: line = "G"+line
                elif not Dfound: line = "D"+line
                elif not Afound: line = "A"+line
                elif not Efound: line = "E"+line
                 
            # check if all lines have been found when starting over again
            if line.startswith("G|") and Gfound: 
                Gfound=False
                Dfound=False
                Afound=False
                Efound=False
    
            # we only have standard tuning and tuning a half step lower. Drop D is ignored
            elif line.startswith(("G|")):
                G_new, _, Gisempty = parse_tab_line(line[0:])
                Gfound=True
            elif line.startswith(("D|")):
                D_new, _, Disempty = parse_tab_line(line[0:])
                Dfound=True
            elif line.startswith(("A|")):
                A_new, _, Aisempty = parse_tab_line(line[0:])
                Afound=True
            elif line.startswith(("E|")):
                E_new, _, Eisempty = parse_tab_line(line[0:])
                Efound=True
                
            # add the basslines only if all strings have been found
            if Efound and Afound and Dfound and Gfound: 
                # add only the bassline to the data if it contains notes
                if len(E_new)!=0 and len(A_new)!=0 and len(D_new)!=0 and len(G_new)!=0:
                    bassTab.append('GDAE')
                    for i in range(1,min(len(E_new),len(A_new),len(D_new),len(G_new))):
                        bassTab.append(str(G_new[i])+str(D_new[i])+str(A_new[i])+str(E_new[i])) 
                    bassTab.append('<END>')
                
                #reset
                Efound=False
                Afound=False
                Dfound=False
                Gfound=False

        if len(bassTab)>1:
            #print_basstab(bassTab)
            return bassTab
        else: 
            print("Cannot access",url)
            return None

    except:
        print('UNKNOWN ERROR! for',url)
        return None


#%%
def print_basstab(Data):
    '''
    Prints the basstab in readable form

    Parameters
    ----------
    Data : List
        The input data, which is a list of strings in which each string contains one beat.

    '''
    G,D,A,E = '','','',''
    end=False
    
    for line in Data:
        if line=='<END>': # we print per bar
           print(G+'\n'+D+'\n'+A+'\n'+E)
           end=True
           G,D,A,E = '','','',''
        else:
            G+=line[0]
            D+=line[1]
            A+=line[2]
            E+=line[3]
    
    # make sure it prints even without END
    if not end:
        print(G+'\n'+D+'\n'+A+'\n'+E)

#%%
from tqdm import tqdm

if __name__ == "__main__":
    
    # Starting url
    ##TODO right now we only have rock
    # filtered on best rating, bass tab, standard tuning and rock
    #url1 = "https://www.ultimate-guitar.com/explore?genres[]=4&order=rating_desc&page="
    #url2 = "&tuning[]=1&type[]=Bass%20Tabs"
    
    # filtered on best rating, bass tab, standard tuning and no genre to hopefully generate better tabs
    url1 = "https://www.ultimate-guitar.com/explore?order=rating_desc&page="
    url2 = "&tuning[]=1&type[]=Bass Tabs"
    N = 100 # total number of pages to scrape, 100 is maximum
    Dataset = []
    
    for i in tqdm(range(N)):
        url = url1+str(i)+url2
    
        # Send a GET request to the current page
        response = requests.get(url)
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        
        data_content = soup.find('div', {'class': 'js-store'})['data-content']
        data = json.loads(data_content)
        
        songs_dict = data['store']['page']['data']['data']['tabs']
        
        # append all data into one array
        for songs in songs_dict:
            DS = load_bassTab(url=songs['tab_url'])
            if DS is not None: Dataset+=DS
                
                                
                
    #%% save the tokens
    if True:
        import pickle
        
        # Assuming your tokenized inputs are stored in a list called 'tokenized_inputs'
        # Save the list using Pickle
        with open('Dataset.pickle', 'wb') as f:
            pickle.dump(Dataset, f)
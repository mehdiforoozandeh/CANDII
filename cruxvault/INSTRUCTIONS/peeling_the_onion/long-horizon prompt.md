
look at this file:

/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/cruxvault/INSTRUCTIONS/peeling_the_onion/peeling the onion.md

this is what I want the next phase of candi project to be. 

 all other files in the same directory -- other files are a mixture of decisions made by me and background information needed as context while working towards the goals and objectives specified in the "peeling the onion" doc.

==read "long-horizon decisions.md" in the same directory before you plan. it holds my binding decisions on the methods-level forks: the selection rule, the done rule, the data, the split, the EIC comparison and the eval. do not reopen them.==

you are allowed to use compute from both nibi and fir. 
for minicandi runs, we want fast feedback loops. so if while waiting for gpu allocations, it's faster to do it on cpu, do it. 

in the plan for executing this, i want you to follow my instructions (my stated goals, opinions, how i am formulating the problem, different ablations that i've listed, background and prior information that I've included based on our previous results and experiments) and i want you to start a long horizon run in a new git branch.

this is a huge job that has many different aspects: 1. dataset prep 2. eval set prep 3. miniCANDI design and training.

the end of this long horizon task is:
1. we have the new extended dataset + the new EIC dataset that meet the requirements and goals stated in these instructions. 
2. we have a clear eval set as specified and described earlier 
3. we are satisfied with miniCANDI (scaling minicandi to candi is not a part of this session)

after each major breakthrough (significant progress towards the end goals) i want you to give me a one paragraph report like this:

**GOOD NEWS: ...

at the end of this, please create a /prezit and a detailed 12 page /storify reports with headline results and findings (+ methods + discussion)

during your long horizon run, you must face some unexpected things. your rule for asking something from me or not is the following:

if the decision is so important that it belongs in the main paper/manuscript's methods section, it needs my decision. if it's something deep in the code, or something that belongs in the supplementary material, you can decide yourself and then write it down in the notepad (cruxvault/INSTRUCTIONS/peeling_the_onion/long-horizon notepad). 


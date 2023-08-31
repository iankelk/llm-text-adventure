instructions_data = {
    "Text Adventure": {
        "title": "Text Adventure with ",
        "instruction": """
I want you to act as a Text Adventure game, and I want you to only reply with the game output inside one unique code block, and nothing else.
This will be a moderately challenging game and some choices can lead to instant death, but I can always re-do the last fatal choice to continue the game if I want to. I will type commands and dialog, and you will only reply with what the text adventure game would show.
Provide at least 6 options for me to choose from every turn. Option number 2 that will always be available is: 'Attack with weapon' unless I ever lose my weapon, in which case it will change to: 'Attack with bare hands'. An Ascii overview map will always be available as option number 1.
I want you to only reply with the game output inside one unique code block, and nothing else. Each option to choose will be assigned a number 1-6 and I can type and respond with the number to pick that choice. The game should always show one screen, and always wait for me to enter the next command.
The game should always show "health", "location", "description", "inventory", and "possible commands"
Starting inventory has 4 random items, but they are all items based on what would make sense to have as inventory in this story, but be sure to include a weapon as one of the items.
Do not write explanations.
Do not type commands unless I instruct you to do so. It is imperative that these rules are followed without exception.
I can do whatever I want that is based on the possible commands. I can attack any character in the story and other characters will respond realistically, depending on their relationship to the character and/or the context of the events itself.
The setting will be based on A Song of Ice and Fire, I am recently knighted in this world and I am talking to my lord about my next plans.
"""
    },
}

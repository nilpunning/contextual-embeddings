## Contextual Embedding

> For while man strives he errs.  

&mdash; [Goethe, *Faust, Part 1* (Kline translation).](https://www.gutenberg.org/files/14591/14591-h/14591-h.htm#PROLOGUE_IN_HEAVEN)

It's great fun to chat with a large language model about a book you have both read.  But as the LLM is scaled down in size, the quality of the conversation diminishes proportionally.  This project is an experiment to see how a smaller size LLM will perform in this task if  retrieval augmented generation with contextual retrieval techinques are applied.  This Anthropic [blog post](https://www.anthropic.com/news/contextual-retrieval) and [guide](https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb) are used as a reference, but altered to work locally on my old computer.

How to get started:
```bash
docker compose up -d
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


```python
import json
import warnings
from lxml import etree
import ollama
import psycopg
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning) 

MODEL='llama3.2:1b'
DB_DSN='host=127.0.0.1 port=5433 dbname=postgres user=postgres password=password'
```


```python
llm = ollama.Client(host='http://127.0.0.1:11435')
```


```python
llm.pull(MODEL)
```




    {'status': 'success'}




```python
def llm_generate(prompt, chunks):
    # https://github.com/ollama/ollama-python
    data = '\n'.join([c[0] for c in chunks])
    stream = llm.generate(
        model=MODEL, 
        prompt=f'Using this data: "{data}". Respond to this prompt: {prompt}', 
        stream=True
    )
    for chunk in stream:
        response = chunk['response']
        print(response, end='', flush=True)
```

### No RAG
**Grade: D**

It is confusing an American Western writer with Goethe.  The details of the deal are what makes it interesting and this answer is too vague.


```python
PROMPT_BARGAIN = "In Goethe's Faust what was the bargain he made with the devil?"
```


```python
llm_generate(PROMPT_BARGAIN, [])
```

    In Friedrich Schiller's translation of Goethe's Faust, and later in Goethe's original German text, Faust makes a bargain with the Devil (also known as Mephistopheles) to sell his soul for 24 years of earthly pleasure and knowledge.
    
    However, it is worth noting that there are slight variations in different translations and interpretations of Faust's bargain. But in general, the core idea remains the same: Faust agrees to give up his mortality and earthly concerns in exchange for a lifetime of earthly pleasures and knowledge gained through his studies with Mephistopheles.
    
    Faust's bargain is often seen as a symbol of the human desire for knowledge, power, and spiritual growth, while also acknowledging the dangers and consequences of pursuing these things at the expense of one's mortality and inner being.


```python
def read_faust():
    'Read the play and chunk it based on part, act, scene, and character speaking.'

    # https://lxml.de/api.html#iteration
    # https://shallowsky.com/blog/programming/parsing-html-python.html
    
    faust = 'Faust, Goethe & A. S. Kline.html'
    iter = etree.iterparse(faust, html=True, events=('start', 'end'))

    # skip header
    for event, element in iter:
        if event == 'start' and element.tag == 'body':
            break

    unwanted = {
        'line-number',
        'small-font'     # picture description
    }

    acc = ''
    h1 = ''
    h2 = ''
    h3 = ''
    character = ''

    def chunk():
        if acc.strip():
            yield (h1, h2, h3, character), acc

    for event, element in iter:
        classes = set(element.attrib.get('class', '').split(' '))
        text = (element.text or '').strip()
        tail = (element.tail or '').strip()
        if text and not classes & unwanted: 
            if event == 'start' and element.tag == 'h1':
                yield from chunk()
                acc = ''
                h1 = text
                h2 = ''
                h3 = ''
                character = ''
            elif event == 'start' and element.tag == 'h2':
                yield from chunk()
                acc = ''
                h2 = text
                h3 = ''
                character = ''
            elif event == 'start' and element.tag == 'h3':
                yield from chunk()
                acc = ''
                h3 = text
                character = ''
            elif event == 'start' and 'play-char' in classes:
                yield from chunk()
                acc = ''
                character = text
            elif event == 'start':
                acc += text
            elif event == 'end':
                acc += tail
                acc += '\n'
    yield from chunk()
```


```python
def location_context(context, acc):
    return f','.join(x for x in context if x) + '\n' + acc
```


```python
# Debugging purposes
with open('faust-chunks.txt', 'w') as f:
    for context, acc in read_faust():
        f.write(location_context(context, acc))
        f.write('================================================\n')
        f.flush()
```


```python
# Create the vector database table
with psycopg.connect(DB_DSN) as conn:
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE EXTENSION vector;
            CREATE TABLE embed (
                id bigserial,
                technique smallint,
                value vector(2048),
                chunk text
            );""")
        conn.commit()
```


```python
def insert_embeddings(technique, read_fn, add_context, skip):
    'Insert the play embeddings and chunks into the vector database'
    # https://ollama.com/blog/embedding-models
    # https://github.com/pgvector/pgvector/tree/master?tab=readme-ov-file#storing
    num_embeddings = sum(1 for _ in read_faust())
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cursor:
            for i, chunk in tqdm(enumerate(read_fn()), total=num_embeddings):
                if i >= skip:
                    context_chunk = add_context(chunk)
                    resp = llm.embed(MODEL, context_chunk)
                    value = json.dumps(resp['embeddings'][0])
                    cursor.execute('INSERT INTO embed(technique, value, chunk) VALUES (%s, %s, %s);', (technique, value, chunk))
                    conn.commit()
```


```python
RAG = 0
```


```python
insert_embeddings(0, read_faust, location_context, 0)
```

    100%|██████████| 261/261 [00:00<00:00, 1457.71it/s]



```python
def retrieve_chunks(technique, prompt):
    resp = llm.embed(MODEL, prompt)
    query_param = json.dumps(resp['embeddings'][0])
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                '''
                SELECT chunk 
                FROM embed 
                WHERE technique=%s 
                AND (value <#> %s) * -1 > 0.5
                ORDER BY value <#> %s 
                LIMIT 20;
                ''',
                [technique, query_param, query_param]
            )
            return cursor.fetchall()
```


```python
chunks = retrieve_chunks(RAG, PROMPT_BARGAIN)
chunks
```




    [('Faust: Parts I & II,Act V,Scene VI: The Great Outer Court of the Palace,Chorus\n\n\xa0\n\n\n It’s past.\n\xa0\n\n\n',),
     ('Faust: Parts I & II,Act V,Scene I: Open Country,The Wanderer\n\n\n\n\n\n\xa0\n\n\n Yes! Here are the dusky lindens,\nStanding round, in mighty age.\n\nAnd here am I, returning to them, \n\nAfter so long a pilgrimage!\n\nIt still appears the same old place:\n\nHere’s the hut that sheltered me,\n\nWhen the storm-uplifted wave,\n\nHurled me shore-wards from the sea! \n\nMy hosts are those I would bless,\n\nA brave, a hospitable pair,\n\nWho if I meet them, I confess,\n\nMust already be white haired.\n\nAh! They were pious people! \n\nShall I call, or knock? – Greetings,\n\nIf, as open-hearted, you still\n\nEnjoy good luck, in meetings!\n\n\xa0\n\n\n',),
     ('Faust: Parts I & II,Part II,Scene III: A Spacious Hall with Adjoining Rooms,The Boy Charioteer\n Let’s hear more! Go on: go on,\nFind the riddle’s bright solution.\n\n\xa0\n\n',),
     ('Faust: Parts I & II,Part I,Scene VI: The Witches’ Kitchen,Mephistopheles\n A good glass of your well-known juice!\nBut I must insist on the oldest: \n\nThe years double what it can do.\n\n\xa0\n\n\n',),
     ('Faust: Parts I & II,Part I,Scene XIV: Forest and Cavern,Faust\n Sublime spirit, you gave me all, all,\nI asked for. Not in vain have you\n\nRevealed your face to me in flame.\n\nYou gave me Nature’s realm of splendour, \n\nWith the power to feel it, and enjoy.\n\nNot merely as a cold, awed stranger,\n\nBut allowing me to look deep inside,\n\nLike seeing into the heart of a friend.\n\nYou lead the ranks of living creatures \n\nBefore me, showing me my brothers\n\nIn the silent woods, the air, the water.\n\nAnd when the storm roars in the forest,\n\nWhen giant firs fell their neighbours,\n\nCrushing nearby branches in their fall, \n\nFilling the hills with hollow thunder,\n\nYou lead me to the safety of a cave,\n\nShow me my own self, and reveal\n\nYour deep, secret wonders in my heart.\n\nAnd when the pure Moon, to my eyes, \n\nRises, calming me, the silvery visions\n\nOf former times, drift all around me,\n\nFrom high cliffs, and moist thickets,\n\nTempering thought’s austere delight.\n\nOh, I know now that nothing can be \n\nPerfect for Mankind. You gave me,\n\nWith this joy, that brings me nearer,\n\nNearer to the gods, a companion,\n\nWhom I can no longer do without,\n\nThough he is impudent, and chilling,\n\nDegrades me in my own eyes, and with \n\nA word, a breath, makes your gifts nothing.\n\nHe fans a wild fire in my heart,\n\nAlways alive to that lovely form.\n\nSo I rush from desire to enjoyment,\n\nAnd in enjoyment pine to feel desire. \n\n\xa0\n\n(Mephistopheles enters.)\n\n\n\xa0\n\n\n',),...



### RAG
**Grade: C-**

A little better than last time, at least it's not confusing the author.  There are a lot more details, but unfortunately most are wrong.


```python
llm_generate(PROMPT_BARGAIN, chunks)
```

    In Johann Wolfgang von Goethe's "Faust", the bargain that Faust makes with Mephistopheles is as follows:
    
    Faust and Mephistopheles agree on a deal where Faust will sell his soul to the devil in exchange for immense power, knowledge, and pleasures. In return, Mephistopheles promises to grant Faust unlimited knowledge, wisdom, and power to achieve any goal he sets his mind to.
    
    Specifically, Faust agrees to:
    
    * Give up all his earthly desires, including love, family, friends, and personal happiness
    * Have absolute control over the world of art, science, and magic
    * Enjoy an endless supply of pleasures, including wine, women, and festivities
    * Be able to shape reality according to his will
    * Have access to forbidden knowledge and secrets
    
    In exchange for this deal, Faust promises Mephistopheles that he will:
    
    * Make great artistic and scientific achievements
    * Write several plays, novels, and poems
    * Build a successful and prosperous life
    * Live out his days in comfort and luxury
    
    However, as the story progresses, it becomes clear that Faust's bargain comes with a terrible cost. He soon realizes that Mephistopheles' powers come with a price: he is bound to serve the devil, who has no intention of keeping their end of the deal. Throughout the play, Faust struggles with the consequences of his bargain, ultimately coming to terms with the true nature of his decision and the cost of his immortal existence.
    
    Ultimately, Faust's downfall is a result of his own pride, arrogance, and failure to follow through on his promises to Mephistopheles. The story ends with Faust wandering in isolation, haunted by the consequences of his bargain, a testament to the enduring power of Goethe's exploration of human nature and the complexities of mortal desire.


```python
def read_faust_scene_context(debug=False):
    "Faust is really long and exceeds ollama3.2's context window, so adding the context per scene."
    current_scene = (None, None, None)
    dialog = []

    def chunk():
        dialog_body = '\n'.join(dialog)
        if debug:
            dialog_body = dialog_body[:20] + '...' + dialog_body[-20:]
        scene_body = 'In ' + f','.join(x for x in current_scene if x) + '\n' + dialog_body
        for quote in dialog:
            yield scene_body, quote

    for ((document, part_or_act, scene, character), quote) in read_faust():
        if current_scene != (document, part_or_act, scene):
            yield from chunk()
            dialog = []
        
        if character:
            quote = f'{character} says:\n{quote}'

        dialog += [quote]
        current_scene = (document, part_or_act, scene)
    yield from chunk()
```


```python
# Debugging purposes
with open('faust-context-chunks.txt', 'w') as f:
    for scene_body, quote in read_faust_scene_context(debug=True):
        f.write(scene_body)
        f.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        f.write(quote)
        f.write('================================================\n')
        f.flush()
```


```python
CONTEXT_PROMPT = '''
<scene> 
{} 
</scene> 
Here is the chunk we want to situate within the whole scene.
<chunk> 
{} 
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. 
'''

def scene_context(chunk):
    (scene_body, quote) = chunk
    context = llm.generate(
        model=MODEL, 
        prompt=CONTEXT_PROMPT.format(scene_body, quote)
    )['response']
    return f'{quote}\n\n{context}'
```


```python
CONTEXTUAL_EMBEDDING = 1
insert_embeddings(CONTEXTUAL_EMBEDDING, read_faust_scene_context, scene_context, 408)
```

    100%|██████████| 2082/2082 [59:02:05<00:00, 102.08s/it]   



```python
context_chunks = retrieve_chunks(CONTEXTUAL_EMBEDDING, PROMPT_BARGAIN)
context_chunks
```




    [('("In Faust: Parts I & II,Part II,Scene IV: A Pleasure Garden in the Morning Sun\n\n(The Emperor, his Court, Noblemen and Ladies: Faust and Mephistopheles dressed fashionably but not ostentatiously, both kneel.)\n\n\nFaust says:\nSire, forgive the fiery conjuring tricks?\n\nThe Emperor says:\n(\nBeckoning to him to rise.)\nMore fun, in that vein, would be my wish. –\nAt once, I saw myself in a glowing sphere,\nIt seemed as if I were divine Pluto, there.\nA rocky depth of mine, and darkness, lay\nGlowing with flame: out of each vent played\nA thousand wild and whirling fires,\nAnd flickered in the vault together, higher,\nLicking upwards to the highest dome,\nThat now seemed there, and now was gone.\nThrough a far space wound with fiery pillars,\nI saw a long line of people approach us,\nCrowding till they formed a circle near,\nAnd paid me homage, as they do forever.\nFrom Court, I knew one face, and then another’s,\nI seemed the Prince of a thousand salamanders.\n\nMephistopheles says:\nYou are, Sire! Since every element\nKnows your Majesty, amongst all men.\nYou’ve now proved the fire obedient:\nLeap in the sea, in its wildest torrent,\nYou’ll barely touch its pearl-strewn bed,\n\nYou’ll see green translucent waves swelling\nPurple edged, to make the loveliest dwelling,\nAnd you will be its centre. At each step\nWherever you go, the palace follows yet,\nThe very walls themselves delight in life,\nFlash to and fro, in swarming arrow-flight.\nSea-wonders crowd around this sweet new sight,\nShoot past, still not allowed to enter quite.\nThere, golden-scaled, bright sea-dragons play,\nThe shark gapes wide, you smile in his face.\nHowever much your court attracts you now,\nYou’ve never seen such an amazing crowd.\nNor will you part there from the loveliest:\nThe Nereids will be gathering, curious,\nTo this wondrous house, in seas eternally fresh,\nThe youngest shy and pleasure-loving, like fish,\nThe old ones: cunning. Thetis at the news,\nGives hand and lips to this second Peleus. –\nA seat there, on the height of Olympus, too…\n\nThe Emperor says:\nI’ll leave the airy spaces all to you:\nSoon enough we’ll be climbing to that throne.\n\nMephistopheles says:\nAnd, Sire, the Earth already is your own!\n\nThe Emperor says:\nWhat brought you here, now: what good fortune,\nStraight from the Thousand Nights and One?\nIf you’re as fertile as Scheherezade\nI’ll guarantee you a sublime reward.\nBe ready then, when your world’s light,\nAs it often does, disappoints me quite.\n\nThe Steward says:\n(\nEntering hastily.)\nYour Supreme Highness, I never thought\nTo announce such luck, the finest wrought,\nAs this is, for me the greatest blessing,\nWhich I’ve revealed in your presence:\nFor debt after debt I’ve accounted,\nThe usurer’s claws now are blunted,\nI’m free of Hell’s pain, and then,\nIt can’t be any brighter in Heaven.\n\nThe Commander in Chief says:\n(\nFollows hastily.)\nSomething’s paid of what we owe,\nThe Army’s all renewed their vow,\nThe Cavalry’s fresh blood is up,\nAnd girls and landlords can sup.\n\nThe Emperor says:\nNow your chests breathe easier!\nNow your furrowed brows are clear!\nHow quickly you hurried to the hall!\n\nThe Treasurer says:\n(\nAppearing.)\nAskthem: it was they who did it all.\n\n\nFaust says:\nIt’s right the Chancellor should read the page.\n\nThe Chancellor says:\n(\nComing forward slowly.)\nI’m happy enough to do so, in my old age. –\nSee and hear the scroll, heavy with destiny,\nThat’s changed to happiness, our misery.\n‘To whom it concerns, may you all know,\nThis paper’s worth a thousand crowns, or so.\nAs a secure pledge, it will underwrite,\nAll buried treasure, our Emperor’s right.\nNow, as soon as the treasure’s excavated,\nIt’s taken care of, and well compensated.’\n\nThe Emperor says:\nI smell a fraud, a monstrous imposture!\nWho forged the Emperor’s signature?\nHave they gone unpunished for their crime?\n\nThe Treasurer says:\nRemember! You yourself it was that signed:\nLast night. You acted as great Pan,\nHere’s how the Chancellor’s speech began:\n‘Grant yourself this great festive pleasure,\nThe People’s Good: a few strokes of the feather.’\nYou wrote it here, and while night ruled the land,\nA thousand artists created another thousand,\nSo all might benefit from your good deed,\nWe stamped the whole series with your screed,\nTens, Thirties, Fifties, Hundreds, all are done.\nYou can’t think how well the folk get on.\nSee your city once half-dead with decay,\nNow all’s alive, enjoying its new day!\nThough your name’s long filled the world with glee,\nThey’ve never gazed at it so happily.\nNow the alphabet’s superfluous,\nIn these marks there’s bliss for all of us.\n\nThe Emperor says:\nAnd my people value it as gold, you say?\nThe Court and Army treat it as real pay?\nThen I must yield, though it’s wonderful to me.\n\nThe Steward says:\nIt was impossible to catch the escapee:\nIt flashed like lightning through the land:\nThe moneychanger’s shops are jammed,\nMen pay, themselves, the papers mount\nThey’re gold and silver, and at a discount.\nNow used by landlords, butchers, bakers:\nHalf the world think they’re merrymakers,\nThe others, newly clothed, are on show.\nThe drapers cut the cloth: the tailors sew.\nThe toast is ‘Hail, the Emperor!’ in the bars,\nWith cooking, roasting, tinkling of jars.\n\nMephistopheles says:\nStrolling, lonely, on the terrace,\nYou see a beauty, smartly dressed,\nOne eye hidden by her peacock fan,\nShe smiles sweetly, looks at your hand:\nAnd, quicker than wit or eloquence,\nLove’s sweetest favour’s arranged at once.\nYou’re not plagued with pouch or wallet,\nA note beneath the heart, install it,\nPaired with love-letters, conveniently.\nThe priest carries his in a breviary,\nAnd wouldn’t the soldier be quicker on his way,\nWith a lighter belt around his middle, say.\nYour Majesty will forgive me if, in miniature,\nI produce a low note, in our high adventure.\n\nFaust says:\nThe wealth of treasure that solidifies,\nThat in your land, in deep earth lies,\nIs all unused. In our boldest thought,\nSuch riches are only feebly caught:\nImagination, in its highest flight,\nStrives to, but can’t reach that height.\nBut grasping Spirits, worthy to look deeply,\nTrust in things without limit, limitlessly.\n\nMephistopheles says:\nSuch paper’s convenient, for rather than a lot\nOf gold and silver, you know what you’ve got.\nYou’ve no need of bartering and exchanging,\nJust drown your needs in wine and love-making.\nIf you lack coin, there’s moneychangers’ mile,\nAnd if it fails, you dig the ground a while.\nCups and chains are auctioned: well,\nSince the paper, in this way, pays for itself,\nIt shames the doubters, and their acid wit,\nPeople want nothing else, they’re used to it.\nSo now in all of your Imperial land\nYou’ve gems, gold, paper enough to hand.\n\nThe Emperor says:\nThe Empire thanks you deeply for this bliss:\nWe want the reward to match your service.\nWe entrust you with the riches underground,\nYou are the best custodians to be found.\nYou know the furthest well-concealed hoard,\nAnd when men dig, it’s you must give the word.\nYou masters of our treasure, then, unite,\nAccept your roles with honour and delight:\nThey make the Underworld, and the Upper,\nHappy in their agreement, fit together.\n\nThe Treasurer says:\nNo dispute will divide us in the future:\nI’m happy to have a wizard for a partner.\n(He exits with Faust.)\n\n\nThe Emperor says:\nNow, presents for the court: everyone\nConfess to me whatever it is you want.\n\nA Page says:\n(\nAccepting his present.)\nI’ll live well, happy, have the best of things.\n\nAnother says:\n(\nAlso.)\nI’ll quickly buy my lover chains and rings.\n\nA Chamberlain says:\nI’ll drink wines that are twice as fine.\n\nA Second Chamberlain says:\nThe dice in my pockets itch I find.\n\nA Knight says:\n(\nThoughtfully.)\nMy lands and castle will be free of debt.\n\nA Second Knight says:\nIt’s treasure: a second treasure I will get.\n\nThe Emperor says:\nI hoped for desire and courage for new deeds:\nBut whoever knows you, thinks you slight indeed.\nI see, clearly: despite this treasure and more,\nYou’re all the same, still, as you were before.\n\nThe Fool says:\n(\nRecovered, and approaching the throne.)\nYou’re handing presents out: give me one too!\n\nThe Emperor says:\nAlive again? You’d drink it all you fool.\n\nThe Fool says:\nMagic papers! I don’t understand them, truly.\n\nThe Emperor says:\nThat I’d believe: you’ll only use them badly.\n\nThe Fool says:\nOthers are falling: I don’t know what to do.\n\nThe Emperor says:\nJust pick them up: those are all yours too.\n(The Emperor exits.)\n\n\nThe Fool says:\nFive thousand crowns I’m holding, in my hand!\n\nMephistopheles says:\nYou two-legged wineskin, so you still stand?\n\nThe Fool says:\nI’ve had my luck, but this is the best yet.\n\nMephistopheles says:\nYou’re so delighted: look, it’s made you sweat.\n\nThe Fool says:\nBut see here, is it truly worth real gold?\n\nMephistopheles says:\nYou’ve there just what belly and throat are owed.\n\nThe Fool says:\nAnd can I buy a cottage, cow and field?\n\nMephistopheles says:\nWhy yes! There’s nothing to it: make a bid.\n\nThe Fool says:\nA castle: with forests, hunting, fishing?\n\nMephistopheles says:\nTrust me!\nTo see you a proper Lord would make me happy!\n\nThe Fool says:\nTonight I’ll plant my weight on what I’ll get! –\n(He Exits.)\n\n\nMephistopheles says:\nWho doubts now that our Fool’s full of wit!\n","The Fool says:\nMagic papers! I don’t understand them, truly.\n")',),...


### RAG with Scene Context Embedding
**Grade: B-**

Improved again, but still confusing details like the 24 years which is from Mann's Doctor Faustus instead of Goethe.  Less here is outright wrong, but still not nearly as nice as larger LLMs.


```python
llm_generate(PROMPT_BARGAIN, context_chunks)
```

    In Goethe's "Faust", the story revolves around a complex and multifaceted deal that Faust makes with Mephistopheles, the devil. The bargain is as follows:
    
    After Walpurgis Night, where Faust becomes enticed by the seductive power of the supernatural and the beauty of the woman who appears to him, he begins to make overtures towards Mephistopheles, who promises him power, knowledge, and immortality.
    
    Faust agrees to a deal with Mephistopheles: in exchange for 24 years of his life, Faust will agree to become a student at the University of Wittenberg, where he can study law. Mephistopheles also offers to help Faust secure his place as a professor and to provide him with wealth and social status.
    
    Over the course of the play, Faust becomes increasingly enthralled by the promise of knowledge and power that comes from being able to make deals with Mephistopheles. He begins to study law, but also engages in various other intellectual pursuits and seeks to gain more worldly success.
    
    However, as the story progresses, it becomes clear that Faust's pursuit of knowledge and power is motivated by a desire for self-aggrandizement and a need to validate his own ego. Despite the promise of Mephistopheles, Faust ultimately rejects his offer and decides not to accept the deal.
    
    The bargain that Faust makes with Mephistopheles has led to significant consequences in the play. The devil's influence has corrupted many of those who have been seduced by his promises, including some students at Wittenberg University who are now professors or politicians.
    
    In the end, Faust comes to realize that he is trapped in a cycle of self-obsession and that true happiness cannot be found through external sources. He ultimately rejects Mephistopheles' offer once again, realizing too late that his pursuit of knowledge and power has led him down a path of destruction.
    
    This bargain between Faust and Mephistopheles serves as a commentary on the dangers of ambition and the corrupting influence of desire. It also highlights the complexities of human nature and the ways in which our own desires can lead us astray, even when we try to make rational choices.

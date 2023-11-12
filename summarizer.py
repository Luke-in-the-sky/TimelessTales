from llm_api import LargeLanguageModelAPI

prompt__summarize_narrative = """
following is a narrative. I want to tell this story to a 5 year old, but the text is too long.
Instead, we want to make a new narrative, where the story is the same but the language is appropriate
for a 5 year old and the story is divided into five to eight chapters, where each chapter is just
long enough for a bedtime story

"""


class StringSplitter:
    """
    A class that provides a method to split a string into chunks based on the longest sequence
    of newline characters ('\\n'). The chunks are created such that each is smaller than or
    equal to a specified maximum chunk length. If a chunk exceeds the maximum length and cannot
    be split on a newline sequence, it is split at the maximum length.
    Usage:
        >>> splitter = StringSplitter(max_chunk_length=45)
        >>> input_string = "Your\\n\\n\\nvery long string\\n\\nwith newline\\ncharacters"
        >>> chunks = splitter.split_string(input_string)
        >>> print(chunks)
        ['Your\\n\\n\\n', 'very long string\\n\\nwith newline\\ncharacters']
    """

    def __init__(self, max_chunk_length):
        if not isinstance(max_chunk_length, int) or max_chunk_length <= 0:
            raise ValueError("max_chunk_length must be a positive integer.")
        self.max_chunk_length = max_chunk_length
        print(f"{self.max_chunk_length=}")

    def split_string(self, input_string):
        """
        Split a string into chunks based on the longest sequence of '\n' characters
        while ensuring each chunk is smaller than max_chunk_length.
        """
        if not isinstance(input_string, str):
            raise ValueError("input_string must be a string.")

        chunks = []
        current_string = input_string
        print(f"-- got input string of len = {len(input_string)}")
        print(f"{self.max_chunk_length=}")
        while len(current_string) > self.max_chunk_length:
            split_index = self._find_split_index(current_string)
            # Split the string at the index and add the left part to chunks
            chunks.append(current_string[:split_index])
            # Update current_string to the remaining part
            current_string = current_string[split_index:]
        # Add the last part if there's any
        if current_string:
            chunks.append(current_string)

        print(f"ended up with chunks of len = {[len(c) for c in chunks]}")

        return chunks

    def _find_split_index(self, string):
        """
        finds the index at which to split `string`
        The crudest way I can think of is just to look for `\n`
        # TODO: consider a better option

        what we want: the index should be
        1. smaller than self.max_chunk_length
        2. as close to self.max_chunk_length as possible, but ideally we also split on semantically meaningful boundaries
        """
        mlen = self.max_chunk_length
        min_len = int(mlen * 0.80)
        print(f"{mlen=}, {min_len=}")

        # if in the last 5% of the string you can find a '\n\n', return that position
        # print(string[min_len:mlen])
        index = string[min_len:mlen].rfind("\n\n") + min_len
        if index > 0:
            return index

        # elif in the last 5% of the string you can find a '\n', return that position
        index = string[min_len:mlen].rfind("\n") + min_len
        if index > 0:
            return index

        # else return the fisrt space from the right
        index = string[:mlen].rfind(" ")
        if index > 0:
            return index

        return self.max_chunk_length - 1


class TextSummarizer:
    """
    A class that encapsulates the functionality required to summarize a long piece of text using a
    large language model, either via an API or a downloaded model from Hugging Face.
    """

    def __init__(
        self,
        language_model: LargeLanguageModelAPI,
        custom_system_prompt: str = None,
        max_length=None,
    ):
        """
        Initializes the TextSummarizer instance with a language model, a system prompt, and a maximum length.
        """
        self.llm = language_model
        self.system_prompt = (
            custom_system_prompt
            if custom_system_prompt
            else prompt__summarize_narrative
            # TODO: we might have a more standard system prompt and concat here
            # the custom one just to express style preferences
        )
        self.set_max_length(max_length)

    def set_max_length(self, max_length):
        """
        Sets the maximum length of text to summarize and initializes a StringSplitter instance.
        If max_length < length of the text to be summarize, we will chunk things down into sizes of max_length
        """
        self.max_length = max_length
        self.string_splitter = StringSplitter(max_length)

    def summarize(self, text: str) -> str:
        """
        Summarizes the provided text using the language model.

        If the concatenated system prompt and text exceeds the maximum length, the text is split into chunks
        using the `StringSplitter.split_text` method. Each chunk is then summarized individually, and the results
        are concatenated to form the final summary.
        """
        # Concatenate the prompt with the text
        full_text = self.llm.compose_prompt(
            message=text, system_message=self.system_prompt
        )

        # Check if the length of the text is within the maximum length of chuncks we want to summarize
        if len(text) <= self.max_length:
            return self.llm.infer(
                full_text
            )  # infer on `full_text`, but have the if on `text`
        else:
            print(f"Text length {len(text)} exceeds maximum length {self.max_length}")
            # If the text exceeds the limit, split it into manageable chunks
            chunks = self.string_splitter.split_string(text)
            full_chunks = [
                self.llm.compose_prompt(
                    message=chunk, system_message=self.system_prompt
                )
                for chunk in chunks
            ]

            # Summarize each chunk individually, then combine them into the final summary
            summaries = [self.llm.infer(full) for full in full_chunks]
            return " ".join(summaries)

    # TODO: we might want to add a method to expand summaries abstractively a little bit, so that
    # `summarize` gives the outline, but the new method expands on individual chapters a bit more


t = """

gossamer fabric over her shoulders, fastening it with a golden girdle
round her waist, and she covered her head with a mantle. Then I went
about among the men everywhere all over the house, and spoke kindly to
each of them man by man: ‘You must not lie sleeping here any longer,’
said I to them, ‘we must be going, for Circe has told me all about it.’
And on this they did as I bade them.

“Even so, however, I did not get them away without misadventure. We had
with us a certain youth named Elpenor, not very remarkable for sense or
courage, who had got drunk and was lying on the house-top away from the
rest of the men, to sleep off his liquor in the cool. When he heard the
noise of the men bustling about, he jumped up on a sudden and forgot
all about coming down by the main staircase, so he tumbled right off
the roof and broke his neck, and his soul went down to the house of
Hades.

“When I had got the men together I said to them, ‘You think you are
about to start home again, but Circe has explained to me that instead
of this, we have got to go to the house of Hades and Proserpine to
consult the ghost of the Theban prophet Teiresias.’

“The men were broken-hearted as they heard me, and threw themselves on
the ground groaning and tearing their hair, but they did not mend
matters by crying. When we reached the sea shore, weeping and lamenting
our fate, Circe brought the ram and the ewe, and we made them fast hard
by the ship. She passed through the midst of us without our knowing it,
for who can see the comings and goings of a god, if the god does not
wish to be seen?




BOOK XI


THE VISIT TO THE DEAD.88


“Then, when we had got down to the sea shore we drew our ship into the
water and got her mast and sails into her; we also put the sheep on
board and took our places, weeping and in great distress of mind.
Circe, that great and cunning goddess, sent us a fair wind that blew
dead aft and staid steadily with us keeping our sails all the time well
filled; so we did whatever wanted doing to the ship’s gear and let her
go as the wind and helmsman headed her. All day long her sails were
full as she held her course over the sea, but when the sun went down
and darkness was over all the earth, we got into the deep waters of the
river Oceanus, where lie the land and city of the Cimmerians who live
enshrouded in mist and darkness which the rays of the sun never pierce
neither at his rising nor as he goes down again out of the heavens, but
the poor wretches live in one long melancholy night. When we got there
we beached the ship, took the sheep out of her, and went along by the
waters of Oceanus till we came to the place of which Circe had told us.

“Here Perimedes and Eurylochus held the victims, while I drew my sword
and dug the trench a cubit each way. I made a drink-offering to all the
dead, first with honey and milk, then with wine, and thirdly with
water, and I sprinkled white barley meal over the whole, praying
earnestly to the poor feckless ghosts, and promising them that when I
got back to Ithaca I would sacrifice a barren heifer for them, the best
I had, and would load the pyre with good things. I also particularly
promised that Teiresias should have a black sheep to himself, the best
in all my flocks. When I had prayed sufficiently to the dead, I cut the
throats of the two sheep and let the blood run into the trench, whereon
the ghosts came trooping up from Erebus—brides,89 young bachelors, old
men worn out with toil, maids who had been crossed in love, and brave
men who had been killed in battle, with their armour still smirched
with blood; they came from every quarter and flitted round the trench
with a strange kind of screaming sound that made me turn pale with
fear. When I saw them coming I told the men to be quick and flay the
carcasses of the two dead sheep and make burnt offerings of them, and
at the same time to repeat prayers to Hades and to Proserpine; but I
sat where I was with my sword drawn and would not let the poor feckless
ghosts come near the blood till Teiresias should have answered my
questions.

“The first ghost that came was that of my comrade Elpenor, for he had
not yet been laid beneath the earth. We had left his body unwaked and
unburied in Circe’s house, for we had had too much else to do. I was
very sorry for him, and cried when I saw him: ‘Elpenor,’ said I, ‘how
did you come down here into this gloom and darkness? You have got here
on foot quicker than I have with my ship.’

“‘Sir,’ he answered with a groan, ‘it was all bad luck, and my own
unspeakable drunkenness. I was lying asleep on the top of Circe’s
house, and never thought of coming down again by the great staircase
but fell right off the roof and broke my neck, so my soul came down to
the house of Hades. And now I beseech you by all those whom you have
left behind you, though they are not here, by your wife, by the father
who brought you up when you were a child, and by Telemachus who is the
one hope of your house, do what I shall now ask you. I know that when
you leave this limbo you will again hold your ship for the Aeaean
island. Do not go thence leaving me unwaked and unburied behind you, or
I may bring heaven’s anger upon you; but burn me with whatever armour I
have, build a barrow for me on the sea shore, that may tell people in
days to come what a poor unlucky fellow I was, and plant over my grave
the oar I used to row with when I was yet alive and with my messmates.’
And I said, ‘My poor fellow, I will do all that you have asked of me.’

“Thus, then, did we sit and hold sad talk with one another, I on the
one side of the trench with my sword held over the blood, and the ghost
of my comrade saying all this to me from the other side. Then came the
ghost of my dead mother Anticlea, daughter to Autolycus. I had left her
alive when I set out for Troy and was moved to tears when I saw her,
but even so, for all my sorrow I would not let her come near the blood
till I had asked my questions of Teiresias.

“Then came also the ghost of Theban Teiresias, with his golden sceptre
in his hand. He knew me and said, ‘Ulysses, noble son of Laertes, why,
poor man, have you left the light of day and come down to visit the
dead in this sad place? Stand back from the trench and withdraw your
sword that I may drink of the blood and answer your questions truly.’

“So I drew back, and sheathed my sword, whereon when he had drank of
the blood he began with his prophecy.

“‘You want to know,’ said he, ‘about your return home, but heaven will
make this hard for you. I do not think that you will escape the eye of
Neptune, who still nurses his bitter grudge against you for having
blinded his son. Still, after much suffering you may get home if you
can restrain yourself and your companions when your ship reaches the
Thrinacian island, where you will find the sheep and cattle belonging
to the sun, who sees and gives ear to everything. If you leave these
flocks unharmed and think of nothing but of getting home, you may yet
after much hardship reach Ithaca; but if you harm them, then I forewarn
you of the destruction both of your ship and of your men. Even though
you may yourself escape, you will return in bad plight after losing all
your men, [in another man’s ship, and you will find trouble in your
house, which will be overrun by high-handed people, who are devouring
your substance under the pretext of paying court and making presents to
your wife.

“‘When you get home you will take your revenge on these suitors; and
after you have killed them by force or fraud in your own house, you
must take a well made oar and carry it on and on, till you come to a
country where the people have never heard of the sea and do not even
mix salt with their food, nor do they know anything about ships, and
oars that are as the wings of a ship. I will give you this certain
token which cannot escape your notice. A wayfarer will meet you and
will say it must be a winnowing shovel that you have got upon your
shoulder; on this you must fix the oar in the ground and sacrifice a
ram, a bull, and a boar to Neptune.90 Then go home and offer hecatombs
to all the gods in heaven one after the other. As for yourself, death
shall come to you from the sea, and your life shall ebb away very
gently when you are full of years and peace of mind, and your people
shall bless you. All that I have said will come true].’91

“‘This,’ I answered, ‘must be as it may please heaven, but tell me and
tell me and tell me true, I see my poor mother’s ghost close by us; she
is sitting by the blood without saying a word, and though I am her own
son she does not remember me and speak to me; tell me, Sir, how I can
make her know me.’

“‘That,’ said he, ‘I can soon do. Any ghost that you let taste of the
blood will talk with you like a reasonable being, but if you do not let
them have any blood they will go away again.’

“On this the ghost of Teiresias went back to the house of Hades, for
his prophecyings had now been spoken, but I sat still where I was until
my mother came up and tasted the blood. Then she knew me at once and
spoke fondly to me, saying, ‘My son, how did you come down to this
abode of darkness while you are still alive? It is a hard thing for the
living to see these places, for between us and them there are great and
terrible waters, and there is Oceanus, which no man can cross on foot,
but he must have a good ship to take him. Are you all this time trying
to find your way home from Troy, and have you never yet got back to
Ithaca nor seen your wife in your own house?’

“‘Mother,’ said I, ‘I was forced to come here to consult the ghost of
the Theban prophet Teiresias. I have never yet been near the Achaean
land nor set foot on my native country, and I have had nothing but one
long series of misfortunes from the very first day that I set out with
Agamemnon for Ilius, the land of noble steeds, to fight the Trojans.
But tell me, and tell me true, in what way did you die? Did you have a
long illness, or did heaven vouchsafe you a gentle easy passage to
eternity? Tell me also about my father, and the son whom I left behind
me, is my property still in their hands, or has some one else got hold
of it, who thinks that I shall not return to claim it? Tell me again
what my wife intends doing, and in what mind she is; does she live with
my son and guard my estate securely, or has she made the best match she
could and married again?’

“My mother answered, ‘Your wife still remains in your house, but she is
in great distress of mind and spends her whole time in tears both night
and day. No one as yet has got possession of your fine property, and
Telemachus still holds your lands undisturbed. He has to entertain
largely, as of course he must, considering his position as a
magistrate,92 and how every one invites him; your father remains at his
old place in the country and never goes near the town. He has no
comfortable bed nor bedding; in the winter he sleeps on the floor in
front of the fire with the men and goes about all in rags, but in
summer, when the warm weather comes on again, he lies out in the
vineyard on a bed of vine leaves thrown any how upon the ground. He
grieves continually about your never having come home, and suffers more
and more as he grows older. As for my own end it was in this wise:
heaven did not take me swiftly and painlessly in my own house, nor was
I attacked by any illness such as those that generally wear people out
and kill them, but my longing to know what you were doing and the force
of my affection for you—this it was that was the death of me.’93

“Then I tried to find some way of embracing my poor mother’s ghost.
Thrice I sprang towards her and tried to clasp her in my arms, but each
time she flitted from my embrace as it were a dream or phantom, and
being touched to the quick I said to her, ‘Mother, why do you not stay
still when I would embrace you? If we could throw our arms around one
another we might find sad comfort in the sharing of our sorrows even in
the house of Hades; does Proserpine want to lay a still further load of
grief upon me by mocking me with a phantom only?’

“‘My son,’ she answered, ‘most ill-fated of all mankind, it is not
Proserpine that is beguiling you, but all people are like this when
they are dead. The sinews no longer hold the flesh and bones together;
these perish in the fierceness of consuming fire as soon as life has
left the body, and the soul flits away as though it were a dream. Now,
however, go back to the light of day as soon as you can, and note all
these things that you may tell them to your wife hereafter.’

“Thus did we converse, and anon Proserpine sent up the ghosts of the
wives and daughters of all the most famous men. They gathered in crowds
about the blood, and I considered how I might question them severally.
In the end I deemed that it would be best to draw the keen blade that
hung by my sturdy thigh, and keep them from all drinking the blood at
once. So they came up one after the other, and each one as I questioned
her told me her race and lineage.

“The first I saw was Tyro. She was daughter of Salmoneus and wife of
Cretheus the son of Aeolus.94 She fell in love with the river Enipeus
who is much the most beautiful river in the whole world. Once when she
was taking a walk by his side as usual, Neptune, disguised as her
lover, lay with her at the mouth of the river, and a huge blue wave
arched itself like a mountain over them to hide both woman and god,
whereon he loosed her virgin girdle and laid her in a deep slumber.
When the god had accomplished the deed of love, he took her hand in his
own and said, ‘Tyro, rejoice in all good will; the embraces of the gods
are not fruitless, and you will have fine twins about this time twelve
months. Take great care of them. I am Neptune, so now go home, but hold
your tongue and do not tell any one.’

“Then he dived under the sea, and she in due course bore Pelias and
Neleus, who both of them served Jove with all their might. Pelias was a
great breeder of sheep and lived in Iolcus, but the other lived in
Pylos. The rest of her children were by Cretheus, namely, Aeson,
Pheres, and Amythaon, who was a mighty warrior and charioteer.

“Next to her I saw Antiope, daughter to Asopus, who could boast of
having slept in the arms of even Jove himself, and who bore him two
sons Amphion and Zethus. These founded Thebes with its seven gates, and
built a wall all round it; for strong though they were they could not
hold Thebes till they had walled it.

“Then I saw Alcmena, the wife of Amphitryon, who also bore to Jove
indomitable Hercules; and Megara who was daughter to great King Creon,
and married the redoubtable son of Amphitryon.

“I also saw fair Epicaste mother of king Oedipodes whose awful lot it
was to marry her own son without suspecting it. He married her after
having killed his father, but the gods proclaimed the whole story to
the world; whereon he remained king of Thebes, in great grief for the
spite the gods had borne him; but Epicaste went to the house of the
mighty jailor Hades, having hanged herself for grief, and the avenging
spirits haunted him as for an outraged mother—to his ruing bitterly
thereafter.

“Then I saw Chloris, whom Neleus married for her beauty, having given
priceless presents for her. She was youngest daughter to Amphion son of
Iasus and king of Minyan Orchomenus, and was Queen in Pylos. She bore
Nestor, Chromius, and Periclymenus, and she also bore that marvellously
lovely woman Pero, who was wooed by all the country round; but Neleus
would only give her to him who should raid the cattle of Iphicles from
the grazing grounds of Phylace, and this was a hard task. The only man
who would undertake to raid them was a certain excellent seer,95 but
the will of heaven was against him, for the rangers of the cattle
caught him and put him in prison; nevertheless when a full year had
passed and the same season came round again, Iphicles set him at
liberty, after he had expounded all the oracles of heaven. Thus, then,
was the will of Jove accomplished.

“And I saw Leda the wife of Tyndarus, who bore him two famous sons,
Castor breaker of horses, and Pollux the mighty boxer. Both these
heroes are lying under the earth, though they are still alive, for by a
special dispensation of Jove, they die and come to life again, each one
of them every other day throughout all time, and they have the rank of
gods.

“After her I saw Iphimedeia wife of Aloeus who boasted the embrace of
Neptune. She bore two sons Otus and Ephialtes, but both were short
lived. They were the finest children that were ever born in this world,
and the best looking, Orion only excepted; for at nine years old they
were nine fathoms high, and measured nine cubits round the chest. They
threatened to make war with the gods in Olympus, and tried to set Mount
Ossa on the top of Mount Olympus, and Mount Pelion on the top of Ossa,
that they might scale heaven itself, and they would have done it too if
they had been grown up, but Apollo, son of Leto, killed both of them,
before they had got so much as a sign of hair upon their cheeks or
chin.

“Then I saw Phaedra, and Procris, and fair Ariadne daughter of the
magician Minos, whom Theseus was carrying off from Crete to Athens, but
he did not enjoy her, for before he could do so Diana killed her in the
island of Dia on account of what Bacchus had said against her.

“I also saw Maera and Clymene and hateful Eriphyle, who sold her own
husband for gold. But it would take me all night if I were to name
every single one of the wives and daughters of heroes whom I saw, and
it is time for me to go to bed, either on board ship with my crew, or
here. As for my escort, heaven and yourselves will see to it.”

Here he ended, and the guests sat all of them enthralled and speechless
throughout the covered cloister. Then Arete said to them:—

“What do you think of this man, O Phaeacians? Is he not tall and good
looking, and is he not clever? True, he is my own guest, but all of you
share in the distinction. Do not be in a hurry to send him away, nor
niggardly in the presents you make to one who is in such great need,
for heaven has blessed all of you with great abundance.”

Then spoke the aged hero Echeneus who was one of the oldest men among
them, “My friends,” said he, “what our august queen has just said to us
is both reasonable and to the purpose, therefore be persuaded by it;
but the decision whether in word or deed rests ultimately with King
Alcinous.”

“The thing shall be done,” exclaimed Alcinous, “as surely as I still
live and reign over the Phaeacians. Our guest is indeed very anxious to
get home, still we must persuade him to remain with us until to-morrow,
by which time I shall be able to get together the whole sum that I mean
to give him. As regards his escort it will be a matter for you all, and
mine above all others as the chief person among you.”

And Ulysses answered, “King Alcinous, if you were to bid me to stay
here for a whole twelve months, and then speed me on my way, loaded
with your noble gifts, I should obey you gladly and it would redound
greatly to my advantage, for I should return fuller-handed to my own
people, and should thus be more respected and beloved by all who see me
when I get back to Ithaca.”

“Ulysses,” replied Alcinous, “not one of us who sees you has any idea
that you are a charlatan or a swindler. I know there are many people
going about who tell such plausible stories that it is very hard to see
through them, but there is a style about your language which assures me
of your good disposition. Moreover you have told the story of your own
misfortunes, and those of the Argives, as though you were a practiced
bard; but tell me, and tell me true, whether you saw any of the mighty
heroes who went to Troy at the same time with yourself, and perished
there. The evenings are still at their longest, and it is not yet bed
time—go on, therefore, with your divine story, for I could stay here
listening till tomorrow morning, so long as you will continue to tell
us of your adventures.”

“Alcinous,” answered Ulysses, “there is a time for making speeches, and
a time for going to bed; nevertheless, since you so desire, I will not
refrain from telling you the still sadder tale of those of my comrades
who did not fall fighting with the Trojans, but perished on their
return, through the treachery of a wicked woman.

“When Proserpine had dismissed the female ghosts in all directions, the
ghost of Agamemnon son of Atreus came sadly up to me, surrounded by
those who had perished with him in the house of Aegisthus. As soon as
he had tasted the blood, he knew me, and weeping bitterly stretched out
his arms towards me to embrace me; but he had no strength nor substance
any more, and I too wept and pitied him as I beheld him. ‘How did you
come by your death,’ said I, ‘King Agamemnon? Did Neptune raise his
winds and waves against you when you were at sea, or did your enemies
make an end of you on the main land when you were cattle-lifting or
sheep-stealing, or while they were fighting in defence of their wives
and city?’

“‘Ulysses,’ he answered, ‘noble son of Laertes, I was not lost at sea
in any storm of Neptune’s raising, nor did my foes despatch me upon the
mainland, but Aegisthus and my wicked wife were the death of me between
them. He asked me to his house, feasted me, and then butchered me most
miserably as though I were a fat beast in a slaughter house, while all
around me my comrades were slain like sheep or pigs for the wedding
breakfast, or picnic, or gorgeous banquet of some great nobleman. You
must have seen numbers of men killed either in a general engagement, or
in single combat, but you never saw anything so truly pitiable as the
way in which we fell in that cloister, with the mixing bowl and the
loaded tables lying all about, and the ground reeking with our blood. I
heard Priam’s daughter Cassandra scream as Clytemnestra killed her
close beside me. I lay dying upon the earth with the sword in my body,
and raised my hands to kill the slut of a murderess, but she slipped
away from me; she would not even close my lips nor my eyes when I was
dying, for there is nothing in this world so cruel and so shameless as
a woman when she has fallen into such guilt as hers was. Fancy
murdering her own husband! I thought I was going to be welcomed home by
my children and my servants, but her abominable crime has brought
disgrace on herself and all women who shall come after—even on the good
ones.’

“And I said, ‘In truth Jove has hated the house of Atreus from first to
last in the matter of their women’s counsels. See how many of us fell
for Helen’s sake, and now it seems that Clytemnestra hatched mischief
against you too during your absence.’

“‘Be sure, therefore,’ continued Agamemnon, ‘and not be too friendly
even with your own wife. Do not tell her all that you know perfectly
well yourself. Tell her a part only, and keep your own counsel about
the rest. Not that your wife, Ulysses, is likely to murder you, for
Penelope is a very admirable woman, and has an excellent nature. We
left her a young bride with an infant at her breast when we set out for
Troy. This child no doubt is now grown up happily to man’s estate,96
and he and his father will have a joyful meeting and embrace one
another as it is right they should do, whereas my wicked wife did not
even allow me the happiness of looking upon my son, but killed me ere I
could do so. Furthermore I say—and lay my saying to your heart—do not
tell people when you are bringing your ship to Ithaca, but steal a
march upon them, for after all this there is no trusting women. But now
tell me, and tell me true, can you give me any news of my son Orestes?
Is he in Orchomenus, or at Pylos, or is he at Sparta with Menelaus—for
I presume that he is still living.’

“And I said, ‘Agamemnon, why do you ask me? I do not know whether your
son is alive or dead, and it is not right to talk when one does not
know.’

“As we two sat weeping and talking thus sadly with one another the
ghost of Achilles came up to us with Patroclus, Antilochus, and Ajax
who was the finest and goodliest man of all the Danaans after the son
of Peleus. The fleet descendant of Aeacus knew me and spoke piteously,
saying, ‘Ulysses, noble son of Laertes, what deed of daring will you
undertake next, that you venture down to the house of Hades among us
silly dead, who are but the ghosts of them that can labour no more?’

“And I said, ‘Achilles, son of Peleus, foremost champion of the
Achaeans, I came to consult Teiresias, and see if he could advise me
about my return home to Ithaca, for I have never yet been able to get
near the Achaean land, nor to set foot in my own country, but have been
in trouble all the time. As for you, Achilles, no one was ever yet so
fortunate as you have been, nor ever will be, for you were adored by
all us Argives as long as you were alive, and now that you are here you
are a great prince among the dead. Do not, therefore, take it so much
to heart even if you are dead.’
"""

s = StringSplitter(10000)
# chunks =
# s._find_split_index(t)
_ = s.split_string(t)

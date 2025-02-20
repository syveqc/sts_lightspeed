import numpy as np
import h5py
import slaythespire
import tqdm

N = 1_000_000
s0 = np.zeros((N, 214), dtype=np.float32)
s1 = np.zeros((N, 214), dtype=np.float32)
card_ids = np.zeros((N,), dtype=np.int16)

monster_encounters = slaythespire.RLInterface.getImplementedMonsterEncounters();

idx = 0
buffer_filled = False

with tqdm.tqdm(total=N) as pbar:
    while True:
        gc = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, 0, 0)
        gc.generateRandomDeck(12, slaythespire.CharacterClass.IRONCLAD, 0, 8)

        bc = slaythespire.BattleContext()
        encounter = monster_encounters[np.random.randint(0, len(monster_encounters))]
        # print(f'encountering {encounter}')
        bc.init(gc, encounter)

        for i in range(500):
            for i in range(20):

                # print('----------------------------------------------------------------------')
                # slaythespire.RLInterface.prettyPrintStateEmbedding(gc, bc)
                # print('----------------------------------------------------------------------')
                playable_cards = bc.getPlayableCards()

                if len(playable_cards) == 0:
                    # print('ended turn')
                    bc.endTurn()
                    break

                cardToPlay = playable_cards[np.random.randint(0, len(playable_cards))]
                targetableMonsterIds = bc.getTargetableMonsterIds()
                if len(targetableMonsterIds) == 0:
                    # print('ended turn')
                    bc.endTurn()
                    break

                target = targetableMonsterIds[np.random.randint(0, len(targetableMonsterIds))]

                # bc.printHand()
                # print('----------------------------------------------------------------------')
                # print(f'played card {cardToPlay}')
                before_state = slaythespire.RLInterface.getStateEmbedding(gc, bc)
                card_id = cardToPlay.id
                bc.playCard(cardToPlay, target)
                after_state = slaythespire.RLInterface.getStateEmbedding(gc, bc)

                s0[idx] = before_state
                s1[idx] = after_state
                card_ids[idx] = card_id
                idx += 1
                pbar.update(1)
                if idx == N:
                    buffer_filled = True

                if bc.outcome != slaythespire.Outcome.UNDECIDED or buffer_filled:
                    break
                # print('----------------------------------------------------------------------')
            if bc.outcome != slaythespire.Outcome.UNDECIDED or buffer_filled:
                # print(bc.outcome)
                break
        if buffer_filled:
            break


with h5py.File('dataset.hdf5', 'w') as f:
    f.create_dataset('s0', data = s0)
    f.create_dataset('s1', data = s1)
    f.create_dataset('card_ids', data = card_ids)

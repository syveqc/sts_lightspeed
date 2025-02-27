//
// Created by keega on 9/16/2021.
//

#include <cstdlib>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

#include <iostream>
#include <random>
#include <sstream>
#include <algorithm>

#include "combat/BattleContext.h"
#include "constants/CharacterClasses.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"
#include "sim/ConsoleSimulator.h"
#include "sim/search/ScumSearchAgent2.h"
#include "sim/SimHelpers.h"
#include "sim/PrintHelpers.h"
#include "game/Game.h"

#include "slaythespire.h"


using namespace sts;

PYBIND11_MODULE(slaythespire, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("play", &sts::py::play, "play Slay the Spire Console");
    m.def("get_seed_str", &SeedHelper::getString, "gets the integral representation of seed string used in the game ui");
    m.def("get_seed_long", &SeedHelper::getLong, "gets the seed string representation of an integral seed");
    m.def("getNNInterface", &sts::NNInterface::getInstance, "gets the NNInterface object");

    pybind11::class_<NNInterface> nnInterface(m, "NNInterface");
    nnInterface.def("getObservation", &NNInterface::getObservation, "get observation array given a GameContext")
        .def("getObservationMaximums", &NNInterface::getObservationMaximums, "get the defined maximum values of the observation space")
        .def_property_readonly("observation_space_size", []() { return NNInterface::observation_space_size; });

    pybind11::class_<RLInterface> rlInterface(m, "RLInterface");
    rlInterface.def("getStateEmbedding", &RLInterface::getStateEmbedding, "get state embedding from given GameContext");
    rlInterface.def("getPlayerGameEmbedding", &RLInterface::getPlayerGameEmbedding, "get player state embedding from given GameContext");
    rlInterface.def("prettyPrintStateEmbedding", &RLInterface::prettyPrintStateEmbedding, "pretty print the state embedding from given GameContext");
    rlInterface.def("getImplementedMonsterEncounters", &RLInterface::getImplementedMonsterEncounters, "return a list of implemented monster encounters");

    pybind11::class_<CardInstance> cardInstance(m, "CardInstance");
    cardInstance.def(pybind11::init<CardId>())
        .def("__repr__", [](const CardInstance &c) {
            std::string s("<slaythespire.CardInstance ");
            s += c.getName();
            if (c.isUpgraded()) {
                s += '+';
                if (c.id == sts::CardId::SEARING_BLOW) {
                    s += std::to_string(c.getUpgradeCount());
                }
            }
            return s += ">";
        }, "returns a string representation of a Card")
        .def("upgrade", &CardInstance::upgrade)
	.def("requires_target", &CardInstance::requiresTarget);

    cardInstance.def_property_readonly("id", &CardInstance::getId)
        .def_property_readonly("upgraded", &CardInstance::isUpgraded)
        .def_property_readonly("upgrade_count", &CardInstance::getUpgradeCount)
        .def_property_readonly("upgradable", &CardInstance::canUpgrade)
        .def_property_readonly("is_strikeCard", &CardInstance::isStrikeCard)
        .def_property_readonly("type", &CardInstance::getType);

    pybind11::class_<search::ScumSearchAgent2> agent(m, "Agent");
    agent.def(pybind11::init<>());
    agent.def_readwrite("simulation_count_base", &search::ScumSearchAgent2::simulationCountBase, "number of simulations the agent uses for monte carlo tree search each turn")
        .def_readwrite("boss_simulation_multiplier", &search::ScumSearchAgent2::bossSimulationMultiplier, "bonus multiplier to the simulation count for boss fights")
        .def_readwrite("pause_on_card_reward", &search::ScumSearchAgent2::pauseOnCardReward, "causes the agent to pause so as to cede control to the user when it encounters a card reward choice")
        .def_readwrite("print_logs", &search::ScumSearchAgent2::printLogs, "when set to true, the agent prints state information as it makes actions")
        .def("playout", &search::ScumSearchAgent2::playout);



    pybind11::class_<BattleContext> battleContext(m, "BattleContext");
    battleContext.def(pybind11::init<>());
    battleContext.def("init", [](BattleContext &bc, GameContext gc, MonsterEncounter encounter){bc.init(gc, encounter);}, "initializes the BattleContext");
    battleContext.def("printMonsterGroup", [](BattleContext &bc){
            for (int i = 0; i < 5; i++) {
                std::cout << (int)bc.monsters.arr[i].id << ": " << bc.monsters.arr[i].curHp << "/" << bc.monsters.arr[i].maxHp << std::endl;
            }
        }, "prints the current monsters in the monster group")
	.def("get_alive_monsters", [](BattleContext &bc){
	    std::vector<MonsterId> monsters;
            for (int i = 0; i < 5; i++) {
	    	if (bc.monsters.arr[i].id != MonsterId::INVALID && bc.monsters.arr[i].curHp > 0)
		    monsters.push_back(bc.monsters.arr[i].id);
            }
	    return monsters;
        }, "returns the monsters from the current monster group")
    	.def("get_alive_monster_intentions", [](BattleContext &bc) {
	    std::vector<MonsterMoveId> moves;
            for (int i = 0; i < 5; i++) {
	    	if (bc.monsters.arr[i].id != MonsterId::INVALID && bc.monsters.arr[i].curHp > 0)
		    moves.push_back(bc.monsters.arr[i].moveHistory[0]);
            }
	    return moves;
	}, "returns the current monster intentions, i.e. the moves the monsters will use after ending the turn")
        .def_readwrite("outcome", &BattleContext::outcome)
        .def("getAliveMonsterIds", [](BattleContext &bc){
             std::vector<int> ids;
             for(int i =0; i < 5; i++)
                if (bc.monsters.arr[i].id != MonsterId::INVALID && bc.monsters.arr[i].isAlive())
                    ids.push_back(i);
             return ids;
             }, "get the ids of all the monsters currently alive")
        .def("getTargetableMonsterIds", [](BattleContext &bc){
             std::vector<int> ids;
             for(int i =0; i < 5; i++)
                if (bc.monsters.arr[i].id != MonsterId::INVALID && bc.monsters.arr[i].isTargetable())
                    ids.push_back(i);
             return ids;
             }, "get the ids of all the monsters currently targetable")
        .def("playCard", [](BattleContext &bc, CardInstance card, int target){
             bc.setState(InputState::EXECUTING_ACTIONS);
             bc.addToBotCard(CardQueueItem(card, target, bc.player.energy));
             bc.executeActions();
        }, "plays a card at a target")
        .def("getCardsInHand", [](BattleContext &bc) {
             std::vector<CardInstance> hand;
             for (int i = 0; i < bc.cards.cardsInHand; i++)
	         hand.push_back(bc.cards.hand[i]);
             return hand;
        }, "get the cards currently in hand")
        .def("getPlayableCards", [](BattleContext &bc) {
             std::vector<CardInstance> playableCards;
             for (int i = 0; i < bc.cards.cardsInHand; i++)
                if (bc.cards.hand[i].canUseOnAnyTarget(bc))
                    playableCards.push_back(bc.cards.hand[i]);
             return playableCards;
        }, "print the cards in the hand")
        .def("printHand", [](BattleContext &bc) {
             for (int i = 0; i < bc.cards.cardsInHand; i++)
                std::cout << bc.cards.hand[i].getName() << std::endl;
        }, "print the cards in the hand")
        .def("playCardInHand", [](BattleContext &bc, int idInHand, int target){
             bc.setState(InputState::EXECUTING_ACTIONS);
             bc.addToBotCard(CardQueueItem(bc.cards.hand[idInHand], target, bc.player.energy));
             bc.executeActions();
        }, "plays a card at a target")
        .def("endTurn", [](BattleContext &bc){
             bc.endTurn();
             bc.setState(InputState::EXECUTING_ACTIONS);
             bc.executeActions();
        }, "ends the players turn")
        .def("canDraw", [](BattleContext &bc) {
        	return bc.cards.drawPile.size() + bc.cards.discardPile.size() > 0;
        }, "returns whether or not there are cards in the draw or discard piles left");

    pybind11::class_<GameContext> gameContext(m, "GameContext");
    gameContext.def(pybind11::init<CharacterClass, std::uint64_t, int>())
        .def("pick_reward_card", &sts::py::pickRewardCard, "choose to obtain the card at the specified index in the card reward list")
        .def("skip_reward_cards", &sts::py::skipRewardCards, "choose to skip the card reward (increases max_hp by 2 with singing bowl)")
        .def("get_card_rewards", [](GameContext &gc, MonsterEncounter encounter){ //changed from original to allow for more flexible retrieval of rewards
            auto r = gc.getRewardsFromBattle(encounter);
            const auto &cardList = r.cardRewards[r.cardRewardCount-1];
            return std::vector<Card>(cardList.begin(), cardList.end());
        }, "return the current card reward list")
        .def_property_readonly("encounter", [](const GameContext &gc) { return gc.info.encounter; })
        .def_property_readonly("deck",
               [](const GameContext &gc) { return std::vector(gc.deck.cards.begin(), gc.deck.cards.end());},
               "returns a copy of the list of cards in the deck"
        )
        .def("obtain_card",
             [](GameContext &gc, Card card) { gc.deck.obtain(gc, card); },
             "add a card to the deck"
        )
        .def("set_player_cur_hp", [](GameContext &gc, int curHp) {
             if (curHp > gc.maxHp)
                curHp = gc.maxHp;
             gc.curHp = curHp;
        }, "sets the players current HP")
        .def("set_player_max_hp", [](GameContext &gc, int maxHp) {
             gc.maxHp = maxHp;
             if (gc.curHp > gc.maxHp)
                gc.curHp = gc.maxHp;
        }, "sets the players maximum HP")
        .def("remove_card",
            [](GameContext &gc, int idx) {
                if (idx < 0 || idx >= gc.deck.size()) {
                    std::cerr << "invalid remove deck remove idx" << std::endl;
                    return;
                }
                gc.deck.remove(gc, idx);
            },
             "remove a card at a idx in the deck"
        )
        .def_property_readonly("relics",
               [] (const GameContext &gc) { return std::vector(gc.relics.relics); },
               "returns a copy of the list of relics"
        )
        .def("__repr__", [](const GameContext &gc) {
            std::ostringstream oss;
            oss << "<" << gc << ">";
            return oss.str();
        }, "returns a string representation of the GameContext")
        .def("populateMonsterList", &GameContext::populateMonsterList, "randomly generate a new encounter and populate the monsters list with it")
        .def("generateRandomDeck", [](GameContext &gc, int numberOfCardsToGenerate, CharacterClass cc, int seed, int retainCards) {
            srand(seed);
            while(gc.deck.size() > retainCards) {
                gc.deck.remove(gc, rand() % gc.deck.size());
            }
            for (int i = 0; i < numberOfCardsToGenerate; i++) {
                std::vector<CardId> *cardCollection;
                switch (cc) {
                    case CharacterClass::IRONCLAD: 
                        cardCollection = &(gc.redCards);
                        break;
                    case CharacterClass::SILENT:
                        cardCollection = &(gc.greenCards);
                        break;
                    case CharacterClass::DEFECT:
                        cardCollection = &(gc.blueCards);
                        break;
                    case CharacterClass::WATCHER:
                        cardCollection = &(gc.purpleCards);
                        break;
                 }
                int size = cardCollection->size() + gc.colorlessCards.size() + gc.curseCards.size();
                int id = rand() % size;

                CardId cardid = CardId::INVALID;
                if (id < cardCollection->size())
                    cardid = (*cardCollection)[id];
                else if(id < cardCollection->size() + gc.colorlessCards.size())
                    cardid = gc.colorlessCards[id-gc.redCards.size()];
                else
                    cardid = gc.colorlessCards[id-gc.redCards.size()-gc.colorlessCards.size()];

                Card card = Card(cardid);
                gc.deck.obtain(gc, card, 1);
            }
        }, "generates a random deck")
        .def("addCardToDeck", [](GameContext &gc, CardId cardId, int count) {
            Card card = Card(cardId);
            gc.deck.obtain(gc, card, count);
        }, "add a specific card to the deck")
        .def("printCardLists", [](GameContext &gc) {
            // Display categorized cards
            std::cout << "Red Cards: ";
            for (const auto& card : gc.redCards) {
                std::cout << "CardId::" << cardEnumStrings[static_cast<int>(card)] << " ";
            }
            std::cout << std::endl;

            std::cout << "Green Cards: ";
            for (const auto& card : gc.greenCards) {
                std::cout << "CardId::" << cardEnumStrings[static_cast<int>(card)] << " ";
            }
            std::cout << std::endl;

            std::cout << "Blue Cards: ";
            for (const auto& card : gc.blueCards) {
                std::cout << "CardId::" << cardEnumStrings[static_cast<int>(card)] << " ";
            }
            std::cout << std::endl;

            std::cout << "Purple Cards: ";
            for (const auto& card : gc.purpleCards) {
                std::cout << "CardId::" << cardEnumStrings[static_cast<int>(card)] << " ";
            }
            std::cout << std::endl;

            std::cout << "Curse Cards: ";
            for (const auto& card : gc.curseCards) {
                std::cout << "CardId::" << cardEnumStrings[static_cast<int>(card)] << " ";
            }
            std::cout << std::endl;

            std::cout << "Colorless Cards: ";
            for (const auto& card : gc.colorlessCards) {
                std::cout << "CardId::" << cardEnumStrings[static_cast<int>(card)] << " ";
            }
            std::cout << std::endl;

        }, "debug method: print all the card lists grouped by color");


    gameContext.def_readwrite("outcome", &GameContext::outcome)
        .def_readwrite("act", &GameContext::act)
        .def_readwrite("floor_num", &GameContext::floorNum)
        .def_readwrite("screen_state", &GameContext::screenState)

        .def_readwrite("seed", &GameContext::seed)
        .def_readwrite("cur_map_node_x", &GameContext::curMapNodeX)
        .def_readwrite("cur_map_node_y", &GameContext::curMapNodeY)
        .def_readwrite("cur_room", &GameContext::curRoom)
//        .def_readwrite("cur_event", &GameContext::curEvent) // todo standardize event names
        .def_readwrite("boss", &GameContext::boss)

        .def_readwrite("cur_hp", &GameContext::curHp)
        .def_readwrite("max_hp", &GameContext::maxHp)
        .def_readwrite("gold", &GameContext::gold)

        .def_readwrite("blue_key", &GameContext::blueKey)
        .def_readwrite("green_key", &GameContext::greenKey)
        .def_readwrite("red_key", &GameContext::redKey)

        .def_readwrite("card_rarity_factor", &GameContext::cardRarityFactor)
        .def_readwrite("potion_chance", &GameContext::potionChance)
        .def_readwrite("monster_chance", &GameContext::monsterChance)
        .def_readwrite("shop_chance", &GameContext::shopChance)
        .def_readwrite("treasure_chance", &GameContext::treasureChance)

        .def_readwrite("shop_remove_count", &GameContext::shopRemoveCount)
        .def_readwrite("speedrun_pace", &GameContext::speedrunPace)
        .def_readwrite("note_for_yourself_card", &GameContext::noteForYourselfCard);

    pybind11::class_<RelicInstance> relic(m, "Relic");
    relic.def_readwrite("id", &RelicInstance::id)
        .def_readwrite("data", &RelicInstance::data);

    pybind11::class_<Map> map(m, "SpireMap");
    map.def(pybind11::init<std::uint64_t, int,int,bool>());
    map.def("get_room_type", &sts::py::getRoomType);
    map.def("has_edge", &sts::py::hasEdge);
    map.def("get_nn_rep", &sts::py::getNNMapRepresentation);
    map.def("__repr__", [](const Map &m) {
        return m.toString(true);
    });

    pybind11::class_<Card> card(m, "Card");
    card.def(pybind11::init<CardId>())
        .def("__repr__", [](const Card &c) {
            std::string s("<slaythespire.Card ");
            s += c.getName();
            if (c.isUpgraded()) {
                s += '+';
                if (c.id == sts::CardId::SEARING_BLOW) {
                    s += std::to_string(c.getUpgraded());
                }
            }
            return s += ">";
        }, "returns a string representation of a Card")
        .def("upgrade", &Card::upgrade)
        .def("getCardId", &Card::getId, "get the id of the Card")
        .def_readwrite("misc", &Card::misc, "value internal to the simulator used for things like ritual dagger damage");

    card.def_property_readonly("id", &Card::getId)
        .def_property_readonly("upgraded", &Card::isUpgraded)
        .def_property_readonly("upgrade_count", &Card::getUpgraded)
        .def_property_readonly("innate", &Card::isInnate)
        .def_property_readonly("transformable", &Card::canTransform)
        .def_property_readonly("upgradable", &Card::canUpgrade)
        .def_property_readonly("is_strikeCard", &Card::isStrikeCard)
        .def_property_readonly("is_starter_strike_or_defend", &Card::isStarterStrikeOrDefend)
        .def_property_readonly("rarity", &Card::getRarity)
        .def_property_readonly("type", &Card::getType);

    pybind11::enum_<GameOutcome> gameOutcome(m, "GameOutcome");
    gameOutcome.value("UNDECIDED", GameOutcome::UNDECIDED)
        .value("PLAYER_VICTORY", GameOutcome::PLAYER_VICTORY)
        .value("PLAYER_LOSS", GameOutcome::PLAYER_LOSS);

    pybind11::enum_<ScreenState> screenState(m, "ScreenState");
    screenState.value("INVALID", ScreenState::INVALID)
        .value("EVENT_SCREEN", ScreenState::EVENT_SCREEN)
        .value("REWARDS", ScreenState::REWARDS)
        .value("BOSS_RELIC_REWARDS", ScreenState::BOSS_RELIC_REWARDS)
        .value("CARD_SELECT", ScreenState::CARD_SELECT)
        .value("MAP_SCREEN", ScreenState::MAP_SCREEN)
        .value("TREASURE_ROOM", ScreenState::TREASURE_ROOM)
        .value("REST_ROOM", ScreenState::REST_ROOM)
        .value("SHOP_ROOM", ScreenState::SHOP_ROOM)
        .value("BATTLE", ScreenState::BATTLE);

    pybind11::enum_<CharacterClass> characterClass(m, "CharacterClass");
    characterClass.value("IRONCLAD", CharacterClass::IRONCLAD)
            .value("SILENT", CharacterClass::SILENT)
            .value("DEFECT", CharacterClass::DEFECT)
            .value("WATCHER", CharacterClass::WATCHER)
            .value("INVALID", CharacterClass::INVALID);

    pybind11::enum_<Room> roomEnum(m, "Room");
    roomEnum.value("SHOP", Room::SHOP)
        .value("REST", Room::REST)
        .value("EVENT", Room::EVENT)
        .value("ELITE", Room::ELITE)
        .value("MONSTER", Room::MONSTER)
        .value("TREASURE", Room::TREASURE)
        .value("BOSS", Room::BOSS)
        .value("BOSS_TREASURE", Room::BOSS_TREASURE)
        .value("NONE", Room::NONE)
        .value("INVALID", Room::INVALID);

    pybind11::enum_<CardRarity>(m, "CardRarity")
        .value("COMMON", CardRarity::COMMON)
        .value("UNCOMMON", CardRarity::UNCOMMON)
        .value("RARE", CardRarity::RARE)
        .value("BASIC", CardRarity::BASIC)
        .value("SPECIAL", CardRarity::SPECIAL)
        .value("CURSE", CardRarity::CURSE)
        .value("INVALID", CardRarity::INVALID);

    pybind11::enum_<CardColor>(m, "CardColor")
        .value("RED", CardColor::RED)
        .value("GREEN", CardColor::GREEN)
        .value("PURPLE", CardColor::PURPLE)
        .value("COLORLESS", CardColor::COLORLESS)
        .value("CURSE", CardColor::CURSE)
        .value("INVALID", CardColor::INVALID);

    pybind11::enum_<CardType>(m, "CardType")
        .value("ATTACK", CardType::ATTACK)
        .value("SKILL", CardType::SKILL)
        .value("POWER", CardType::POWER)
        .value("CURSE", CardType::CURSE)
        .value("STATUS", CardType::STATUS)
        .value("INVALID", CardType::INVALID);

    pybind11::enum_<Outcome>(m, "Outcome")
        .value("UNDECIDED", Outcome::UNDECIDED)
        .value("PLAYER_VICTORY", Outcome::PLAYER_VICTORY)
        .value("PLAYER_LOSS", Outcome::PLAYER_LOSS);

    pybind11::enum_<CardId>(m, "CardId")
        .value("INVALID", CardId::INVALID)
        .value("ACCURACY", CardId::ACCURACY)
        .value("ACROBATICS", CardId::ACROBATICS)
        .value("ADRENALINE", CardId::ADRENALINE)
        .value("AFTER_IMAGE", CardId::AFTER_IMAGE)
        .value("AGGREGATE", CardId::AGGREGATE)
        .value("ALCHEMIZE", CardId::ALCHEMIZE)
        .value("ALL_FOR_ONE", CardId::ALL_FOR_ONE)
        .value("ALL_OUT_ATTACK", CardId::ALL_OUT_ATTACK)
        .value("ALPHA", CardId::ALPHA)
        .value("AMPLIFY", CardId::AMPLIFY)
        .value("ANGER", CardId::ANGER)
        .value("APOTHEOSIS", CardId::APOTHEOSIS)
        .value("APPARITION", CardId::APPARITION)
        .value("ARMAMENTS", CardId::ARMAMENTS)
        .value("ASCENDERS_BANE", CardId::ASCENDERS_BANE)
        .value("AUTO_SHIELDS", CardId::AUTO_SHIELDS)
        .value("A_THOUSAND_CUTS", CardId::A_THOUSAND_CUTS)
        .value("BACKFLIP", CardId::BACKFLIP)
        .value("BACKSTAB", CardId::BACKSTAB)
        .value("BALL_LIGHTNING", CardId::BALL_LIGHTNING)
        .value("BANDAGE_UP", CardId::BANDAGE_UP)
        .value("BANE", CardId::BANE)
        .value("BARRAGE", CardId::BARRAGE)
        .value("BARRICADE", CardId::BARRICADE)
        .value("BASH", CardId::BASH)
        .value("BATTLE_HYMN", CardId::BATTLE_HYMN)
        .value("BATTLE_TRANCE", CardId::BATTLE_TRANCE)
        .value("BEAM_CELL", CardId::BEAM_CELL)
        .value("BECOME_ALMIGHTY", CardId::BECOME_ALMIGHTY)
        .value("BERSERK", CardId::BERSERK)
        .value("BETA", CardId::BETA)
        .value("BIASED_COGNITION", CardId::BIASED_COGNITION)
        .value("BITE", CardId::BITE)
        .value("BLADE_DANCE", CardId::BLADE_DANCE)
        .value("BLASPHEMY", CardId::BLASPHEMY)
        .value("BLIND", CardId::BLIND)
        .value("BLIZZARD", CardId::BLIZZARD)
        .value("BLOODLETTING", CardId::BLOODLETTING)
        .value("BLOOD_FOR_BLOOD", CardId::BLOOD_FOR_BLOOD)
        .value("BLUDGEON", CardId::BLUDGEON)
        .value("BLUR", CardId::BLUR)
        .value("BODY_SLAM", CardId::BODY_SLAM)
        .value("BOOT_SEQUENCE", CardId::BOOT_SEQUENCE)
        .value("BOUNCING_FLASK", CardId::BOUNCING_FLASK)
        .value("BOWLING_BASH", CardId::BOWLING_BASH)
        .value("BRILLIANCE", CardId::BRILLIANCE)
        .value("BRUTALITY", CardId::BRUTALITY)
        .value("BUFFER", CardId::BUFFER)
        .value("BULLET_TIME", CardId::BULLET_TIME)
        .value("BULLSEYE", CardId::BULLSEYE)
        .value("BURN", CardId::BURN)
        .value("BURNING_PACT", CardId::BURNING_PACT)
        .value("BURST", CardId::BURST)
        .value("CALCULATED_GAMBLE", CardId::CALCULATED_GAMBLE)
        .value("CALTROPS", CardId::CALTROPS)
        .value("CAPACITOR", CardId::CAPACITOR)
        .value("CARNAGE", CardId::CARNAGE)
        .value("CARVE_REALITY", CardId::CARVE_REALITY)
        .value("CATALYST", CardId::CATALYST)
        .value("CHAOS", CardId::CHAOS)
        .value("CHARGE_BATTERY", CardId::CHARGE_BATTERY)
        .value("CHILL", CardId::CHILL)
        .value("CHOKE", CardId::CHOKE)
        .value("CHRYSALIS", CardId::CHRYSALIS)
        .value("CLASH", CardId::CLASH)
        .value("CLAW", CardId::CLAW)
        .value("CLEAVE", CardId::CLEAVE)
        .value("CLOAK_AND_DAGGER", CardId::CLOAK_AND_DAGGER)
        .value("CLOTHESLINE", CardId::CLOTHESLINE)
        .value("CLUMSY", CardId::CLUMSY)
        .value("COLD_SNAP", CardId::COLD_SNAP)
        .value("COLLECT", CardId::COLLECT)
        .value("COMBUST", CardId::COMBUST)
        .value("COMPILE_DRIVER", CardId::COMPILE_DRIVER)
        .value("CONCENTRATE", CardId::CONCENTRATE)
        .value("CONCLUDE", CardId::CONCLUDE)
        .value("CONJURE_BLADE", CardId::CONJURE_BLADE)
        .value("CONSECRATE", CardId::CONSECRATE)
        .value("CONSUME", CardId::CONSUME)
        .value("COOLHEADED", CardId::COOLHEADED)
        .value("CORE_SURGE", CardId::CORE_SURGE)
        .value("CORPSE_EXPLOSION", CardId::CORPSE_EXPLOSION)
        .value("CORRUPTION", CardId::CORRUPTION)
        .value("CREATIVE_AI", CardId::CREATIVE_AI)
        .value("CRESCENDO", CardId::CRESCENDO)
        .value("CRIPPLING_CLOUD", CardId::CRIPPLING_CLOUD)
        .value("CRUSH_JOINTS", CardId::CRUSH_JOINTS)
        .value("CURSE_OF_THE_BELL", CardId::CURSE_OF_THE_BELL)
        .value("CUT_THROUGH_FATE", CardId::CUT_THROUGH_FATE)
        .value("DAGGER_SPRAY", CardId::DAGGER_SPRAY)
        .value("DAGGER_THROW", CardId::DAGGER_THROW)
        .value("DARKNESS", CardId::DARKNESS)
        .value("DARK_EMBRACE", CardId::DARK_EMBRACE)
        .value("DARK_SHACKLES", CardId::DARK_SHACKLES)
        .value("DASH", CardId::DASH)
        .value("DAZED", CardId::DAZED)
        .value("DEADLY_POISON", CardId::DEADLY_POISON)
        .value("DECAY", CardId::DECAY)
        .value("DECEIVE_REALITY", CardId::DECEIVE_REALITY)
        .value("DEEP_BREATH", CardId::DEEP_BREATH)
        .value("DEFEND_BLUE", CardId::DEFEND_BLUE)
        .value("DEFEND_GREEN", CardId::DEFEND_GREEN)
        .value("DEFEND_PURPLE", CardId::DEFEND_PURPLE)
        .value("DEFEND_RED", CardId::DEFEND_RED)
        .value("DEFLECT", CardId::DEFLECT)
        .value("DEFRAGMENT", CardId::DEFRAGMENT)
        .value("DEMON_FORM", CardId::DEMON_FORM)
        .value("DEUS_EX_MACHINA", CardId::DEUS_EX_MACHINA)
        .value("DEVA_FORM", CardId::DEVA_FORM)
        .value("DEVOTION", CardId::DEVOTION)
        .value("DIE_DIE_DIE", CardId::DIE_DIE_DIE)
        .value("DISARM", CardId::DISARM)
        .value("DISCOVERY", CardId::DISCOVERY)
        .value("DISTRACTION", CardId::DISTRACTION)
        .value("DODGE_AND_ROLL", CardId::DODGE_AND_ROLL)
        .value("DOOM_AND_GLOOM", CardId::DOOM_AND_GLOOM)
        .value("DOPPELGANGER", CardId::DOPPELGANGER)
        .value("DOUBLE_ENERGY", CardId::DOUBLE_ENERGY)
        .value("DOUBLE_TAP", CardId::DOUBLE_TAP)
        .value("DOUBT", CardId::DOUBT)
        .value("DRAMATIC_ENTRANCE", CardId::DRAMATIC_ENTRANCE)
        .value("DROPKICK", CardId::DROPKICK)
        .value("DUALCAST", CardId::DUALCAST)
        .value("DUAL_WIELD", CardId::DUAL_WIELD)
        .value("ECHO_FORM", CardId::ECHO_FORM)
        .value("ELECTRODYNAMICS", CardId::ELECTRODYNAMICS)
        .value("EMPTY_BODY", CardId::EMPTY_BODY)
        .value("EMPTY_FIST", CardId::EMPTY_FIST)
        .value("EMPTY_MIND", CardId::EMPTY_MIND)
        .value("ENDLESS_AGONY", CardId::ENDLESS_AGONY)
        .value("ENLIGHTENMENT", CardId::ENLIGHTENMENT)
        .value("ENTRENCH", CardId::ENTRENCH)
        .value("ENVENOM", CardId::ENVENOM)
        .value("EQUILIBRIUM", CardId::EQUILIBRIUM)
        .value("ERUPTION", CardId::ERUPTION)
        .value("ESCAPE_PLAN", CardId::ESCAPE_PLAN)
        .value("ESTABLISHMENT", CardId::ESTABLISHMENT)
        .value("EVALUATE", CardId::EVALUATE)
        .value("EVISCERATE", CardId::EVISCERATE)
        .value("EVOLVE", CardId::EVOLVE)
        .value("EXHUME", CardId::EXHUME)
        .value("EXPERTISE", CardId::EXPERTISE)
        .value("EXPUNGER", CardId::EXPUNGER)
        .value("FAME_AND_FORTUNE", CardId::FAME_AND_FORTUNE)
        .value("FASTING", CardId::FASTING)
        .value("FEAR_NO_EVIL", CardId::FEAR_NO_EVIL)
        .value("FEED", CardId::FEED)
        .value("FEEL_NO_PAIN", CardId::FEEL_NO_PAIN)
        .value("FIEND_FIRE", CardId::FIEND_FIRE)
        .value("FINESSE", CardId::FINESSE)
        .value("FINISHER", CardId::FINISHER)
        .value("FIRE_BREATHING", CardId::FIRE_BREATHING)
        .value("FISSION", CardId::FISSION)
        .value("FLAME_BARRIER", CardId::FLAME_BARRIER)
        .value("FLASH_OF_STEEL", CardId::FLASH_OF_STEEL)
        .value("FLECHETTES", CardId::FLECHETTES)
        .value("FLEX", CardId::FLEX)
        .value("FLURRY_OF_BLOWS", CardId::FLURRY_OF_BLOWS)
        .value("FLYING_KNEE", CardId::FLYING_KNEE)
        .value("FLYING_SLEEVES", CardId::FLYING_SLEEVES)
        .value("FOLLOW_UP", CardId::FOLLOW_UP)
        .value("FOOTWORK", CardId::FOOTWORK)
        .value("FORCE_FIELD", CardId::FORCE_FIELD)
        .value("FOREIGN_INFLUENCE", CardId::FOREIGN_INFLUENCE)
        .value("FORESIGHT", CardId::FORESIGHT)
        .value("FORETHOUGHT", CardId::FORETHOUGHT)
        .value("FTL", CardId::FTL)
        .value("FUSION", CardId::FUSION)
        .value("GENETIC_ALGORITHM", CardId::GENETIC_ALGORITHM)
        .value("GHOSTLY_ARMOR", CardId::GHOSTLY_ARMOR)
        .value("GLACIER", CardId::GLACIER)
        .value("GLASS_KNIFE", CardId::GLASS_KNIFE)
        .value("GOOD_INSTINCTS", CardId::GOOD_INSTINCTS)
        .value("GO_FOR_THE_EYES", CardId::GO_FOR_THE_EYES)
        .value("GRAND_FINALE", CardId::GRAND_FINALE)
        .value("HALT", CardId::HALT)
        .value("HAND_OF_GREED", CardId::HAND_OF_GREED)
        .value("HAVOC", CardId::HAVOC)
        .value("HEADBUTT", CardId::HEADBUTT)
        .value("HEATSINKS", CardId::HEATSINKS)
        .value("HEAVY_BLADE", CardId::HEAVY_BLADE)
        .value("HEEL_HOOK", CardId::HEEL_HOOK)
        .value("HELLO_WORLD", CardId::HELLO_WORLD)
        .value("HEMOKINESIS", CardId::HEMOKINESIS)
        .value("HOLOGRAM", CardId::HOLOGRAM)
        .value("HYPERBEAM", CardId::HYPERBEAM)
        .value("IMMOLATE", CardId::IMMOLATE)
        .value("IMPATIENCE", CardId::IMPATIENCE)
        .value("IMPERVIOUS", CardId::IMPERVIOUS)
        .value("INDIGNATION", CardId::INDIGNATION)
        .value("INFERNAL_BLADE", CardId::INFERNAL_BLADE)
        .value("INFINITE_BLADES", CardId::INFINITE_BLADES)
        .value("INFLAME", CardId::INFLAME)
        .value("INJURY", CardId::INJURY)
        .value("INNER_PEACE", CardId::INNER_PEACE)
        .value("INSIGHT", CardId::INSIGHT)
        .value("INTIMIDATE", CardId::INTIMIDATE)
        .value("IRON_WAVE", CardId::IRON_WAVE)
        .value("JAX", CardId::JAX)
        .value("JACK_OF_ALL_TRADES", CardId::JACK_OF_ALL_TRADES)
        .value("JUDGMENT", CardId::JUDGMENT)
        .value("JUGGERNAUT", CardId::JUGGERNAUT)
        .value("JUST_LUCKY", CardId::JUST_LUCKY)
        .value("LEAP", CardId::LEAP)
        .value("LEG_SWEEP", CardId::LEG_SWEEP)
        .value("LESSON_LEARNED", CardId::LESSON_LEARNED)
        .value("LIKE_WATER", CardId::LIKE_WATER)
        .value("LIMIT_BREAK", CardId::LIMIT_BREAK)
        .value("LIVE_FOREVER", CardId::LIVE_FOREVER)
        .value("LOOP", CardId::LOOP)
        .value("MACHINE_LEARNING", CardId::MACHINE_LEARNING)
        .value("MADNESS", CardId::MADNESS)
        .value("MAGNETISM", CardId::MAGNETISM)
        .value("MALAISE", CardId::MALAISE)
        .value("MASTERFUL_STAB", CardId::MASTERFUL_STAB)
        .value("MASTER_OF_STRATEGY", CardId::MASTER_OF_STRATEGY)
        .value("MASTER_REALITY", CardId::MASTER_REALITY)
        .value("MAYHEM", CardId::MAYHEM)
        .value("MEDITATE", CardId::MEDITATE)
        .value("MELTER", CardId::MELTER)
        .value("MENTAL_FORTRESS", CardId::MENTAL_FORTRESS)
        .value("METALLICIZE", CardId::METALLICIZE)
        .value("METAMORPHOSIS", CardId::METAMORPHOSIS)
        .value("METEOR_STRIKE", CardId::METEOR_STRIKE)
        .value("MIND_BLAST", CardId::MIND_BLAST)
        .value("MIRACLE", CardId::MIRACLE)
        .value("MULTI_CAST", CardId::MULTI_CAST)
        .value("NECRONOMICURSE", CardId::NECRONOMICURSE)
        .value("NEUTRALIZE", CardId::NEUTRALIZE)
        .value("NIGHTMARE", CardId::NIGHTMARE)
        .value("NIRVANA", CardId::NIRVANA)
        .value("NORMALITY", CardId::NORMALITY)
        .value("NOXIOUS_FUMES", CardId::NOXIOUS_FUMES)
        .value("OFFERING", CardId::OFFERING)
        .value("OMEGA", CardId::OMEGA)
        .value("OMNISCIENCE", CardId::OMNISCIENCE)
        .value("OUTMANEUVER", CardId::OUTMANEUVER)
        .value("OVERCLOCK", CardId::OVERCLOCK)
        .value("PAIN", CardId::PAIN)
        .value("PANACEA", CardId::PANACEA)
        .value("PANACHE", CardId::PANACHE)
        .value("PANIC_BUTTON", CardId::PANIC_BUTTON)
        .value("PARASITE", CardId::PARASITE)
        .value("PERFECTED_STRIKE", CardId::PERFECTED_STRIKE)
        .value("PERSEVERANCE", CardId::PERSEVERANCE)
        .value("PHANTASMAL_KILLER", CardId::PHANTASMAL_KILLER)
        .value("PIERCING_WAIL", CardId::PIERCING_WAIL)
        .value("POISONED_STAB", CardId::POISONED_STAB)
        .value("POMMEL_STRIKE", CardId::POMMEL_STRIKE)
        .value("POWER_THROUGH", CardId::POWER_THROUGH)
        .value("PRAY", CardId::PRAY)
        .value("PREDATOR", CardId::PREDATOR)
        .value("PREPARED", CardId::PREPARED)
        .value("PRESSURE_POINTS", CardId::PRESSURE_POINTS)
        .value("PRIDE", CardId::PRIDE)
        .value("PROSTRATE", CardId::PROSTRATE)
        .value("PROTECT", CardId::PROTECT)
        .value("PUMMEL", CardId::PUMMEL)
        .value("PURITY", CardId::PURITY)
        .value("QUICK_SLASH", CardId::QUICK_SLASH)
        .value("RAGE", CardId::RAGE)
        .value("RAGNAROK", CardId::RAGNAROK)
        .value("RAINBOW", CardId::RAINBOW)
        .value("RAMPAGE", CardId::RAMPAGE)
        .value("REACH_HEAVEN", CardId::REACH_HEAVEN)
        .value("REAPER", CardId::REAPER)
        .value("REBOOT", CardId::REBOOT)
        .value("REBOUND", CardId::REBOUND)
        .value("RECKLESS_CHARGE", CardId::RECKLESS_CHARGE)
        .value("RECURSION", CardId::RECURSION)
        .value("RECYCLE", CardId::RECYCLE)
        .value("REFLEX", CardId::REFLEX)
        .value("REGRET", CardId::REGRET)
        .value("REINFORCED_BODY", CardId::REINFORCED_BODY)
        .value("REPROGRAM", CardId::REPROGRAM)
        .value("RIDDLE_WITH_HOLES", CardId::RIDDLE_WITH_HOLES)
        .value("RIP_AND_TEAR", CardId::RIP_AND_TEAR)
        .value("RITUAL_DAGGER", CardId::RITUAL_DAGGER)
        .value("RUPTURE", CardId::RUPTURE)
        .value("RUSHDOWN", CardId::RUSHDOWN)
        .value("SADISTIC_NATURE", CardId::SADISTIC_NATURE)
        .value("SAFETY", CardId::SAFETY)
        .value("SANCTITY", CardId::SANCTITY)
        .value("SANDS_OF_TIME", CardId::SANDS_OF_TIME)
        .value("SASH_WHIP", CardId::SASH_WHIP)
        .value("SCRAPE", CardId::SCRAPE)
        .value("SCRAWL", CardId::SCRAWL)
        .value("SEARING_BLOW", CardId::SEARING_BLOW)
        .value("SECOND_WIND", CardId::SECOND_WIND)
        .value("SECRET_TECHNIQUE", CardId::SECRET_TECHNIQUE)
        .value("SECRET_WEAPON", CardId::SECRET_WEAPON)
        .value("SEEING_RED", CardId::SEEING_RED)
        .value("SEEK", CardId::SEEK)
        .value("SELF_REPAIR", CardId::SELF_REPAIR)
        .value("SENTINEL", CardId::SENTINEL)
        .value("SETUP", CardId::SETUP)
        .value("SEVER_SOUL", CardId::SEVER_SOUL)
        .value("SHAME", CardId::SHAME)
        .value("SHIV", CardId::SHIV)
        .value("SHOCKWAVE", CardId::SHOCKWAVE)
        .value("SHRUG_IT_OFF", CardId::SHRUG_IT_OFF)
        .value("SIGNATURE_MOVE", CardId::SIGNATURE_MOVE)
        .value("SIMMERING_FURY", CardId::SIMMERING_FURY)
        .value("SKEWER", CardId::SKEWER)
        .value("SKIM", CardId::SKIM)
        .value("SLICE", CardId::SLICE)
        .value("SLIMED", CardId::SLIMED)
        .value("SMITE", CardId::SMITE)
        .value("SNEAKY_STRIKE", CardId::SNEAKY_STRIKE)
        .value("SPIRIT_SHIELD", CardId::SPIRIT_SHIELD)
        .value("SPOT_WEAKNESS", CardId::SPOT_WEAKNESS)
        .value("STACK", CardId::STACK)
        .value("STATIC_DISCHARGE", CardId::STATIC_DISCHARGE)
        .value("STEAM_BARRIER", CardId::STEAM_BARRIER)
        .value("STORM", CardId::STORM)
        .value("STORM_OF_STEEL", CardId::STORM_OF_STEEL)
        .value("STREAMLINE", CardId::STREAMLINE)
        .value("STRIKE_BLUE", CardId::STRIKE_BLUE)
        .value("STRIKE_GREEN", CardId::STRIKE_GREEN)
        .value("STRIKE_PURPLE", CardId::STRIKE_PURPLE)
        .value("STRIKE_RED", CardId::STRIKE_RED)
        .value("STUDY", CardId::STUDY)
        .value("SUCKER_PUNCH", CardId::SUCKER_PUNCH)
        .value("SUNDER", CardId::SUNDER)
        .value("SURVIVOR", CardId::SURVIVOR)
        .value("SWEEPING_BEAM", CardId::SWEEPING_BEAM)
        .value("SWIFT_STRIKE", CardId::SWIFT_STRIKE)
        .value("SWIVEL", CardId::SWIVEL)
        .value("SWORD_BOOMERANG", CardId::SWORD_BOOMERANG)
        .value("TACTICIAN", CardId::TACTICIAN)
        .value("TALK_TO_THE_HAND", CardId::TALK_TO_THE_HAND)
        .value("TANTRUM", CardId::TANTRUM)
        .value("TEMPEST", CardId::TEMPEST)
        .value("TERROR", CardId::TERROR)
        .value("THE_BOMB", CardId::THE_BOMB)
        .value("THINKING_AHEAD", CardId::THINKING_AHEAD)
        .value("THIRD_EYE", CardId::THIRD_EYE)
        .value("THROUGH_VIOLENCE", CardId::THROUGH_VIOLENCE)
        .value("THUNDERCLAP", CardId::THUNDERCLAP)
        .value("THUNDER_STRIKE", CardId::THUNDER_STRIKE)
        .value("TOOLS_OF_THE_TRADE", CardId::TOOLS_OF_THE_TRADE)
        .value("TRANQUILITY", CardId::TRANQUILITY)
        .value("TRANSMUTATION", CardId::TRANSMUTATION)
        .value("TRIP", CardId::TRIP)
        .value("TRUE_GRIT", CardId::TRUE_GRIT)
        .value("TURBO", CardId::TURBO)
        .value("TWIN_STRIKE", CardId::TWIN_STRIKE)
        .value("UNLOAD", CardId::UNLOAD)
        .value("UPPERCUT", CardId::UPPERCUT)
        .value("VAULT", CardId::VAULT)
        .value("VIGILANCE", CardId::VIGILANCE)
        .value("VIOLENCE", CardId::VIOLENCE)
        .value("VOID", CardId::VOID)
        .value("WALLOP", CardId::WALLOP)
        .value("WARCRY", CardId::WARCRY)
        .value("WAVE_OF_THE_HAND", CardId::WAVE_OF_THE_HAND)
        .value("WEAVE", CardId::WEAVE)
        .value("WELL_LAID_PLANS", CardId::WELL_LAID_PLANS)
        .value("WHEEL_KICK", CardId::WHEEL_KICK)
        .value("WHIRLWIND", CardId::WHIRLWIND)
        .value("WHITE_NOISE", CardId::WHITE_NOISE)
        .value("WILD_STRIKE", CardId::WILD_STRIKE)
        .value("WINDMILL_STRIKE", CardId::WINDMILL_STRIKE)
        .value("WISH", CardId::WISH)
        .value("WORSHIP", CardId::WORSHIP)
        .value("WOUND", CardId::WOUND)
        .value("WRAITH_FORM", CardId::WRAITH_FORM)
        .value("WREATH_OF_FLAME", CardId::WREATH_OF_FLAME)
        .value("WRITHE", CardId::WRITHE)
        .value("ZAP", CardId::ZAP);

    pybind11::enum_<MonsterEncounter> meEnum(m, "MonsterEncounter");
    meEnum.value("INVALID", ME::INVALID)
        .value("CULTIST", ME::CULTIST)
        .value("JAW_WORM", ME::JAW_WORM)
        .value("TWO_LOUSE", ME::TWO_LOUSE)
        .value("SMALL_SLIMES", ME::SMALL_SLIMES)
        .value("BLUE_SLAVER", ME::BLUE_SLAVER)
        .value("GREMLIN_GANG", ME::GREMLIN_GANG)
        .value("LOOTER", ME::LOOTER)
        .value("LARGE_SLIME", ME::LARGE_SLIME)
        .value("LOTS_OF_SLIMES", ME::LOTS_OF_SLIMES)
        .value("EXORDIUM_THUGS", ME::EXORDIUM_THUGS)
        .value("EXORDIUM_WILDLIFE", ME::EXORDIUM_WILDLIFE)
        .value("RED_SLAVER", ME::RED_SLAVER)
        .value("THREE_LOUSE", ME::THREE_LOUSE)
        .value("TWO_FUNGI_BEASTS", ME::TWO_FUNGI_BEASTS)
        .value("GREMLIN_NOB", ME::GREMLIN_NOB)
        .value("LAGAVULIN", ME::LAGAVULIN)
        .value("THREE_SENTRIES", ME::THREE_SENTRIES)
        .value("SLIME_BOSS", ME::SLIME_BOSS)
        .value("THE_GUARDIAN", ME::THE_GUARDIAN)
        .value("HEXAGHOST", ME::HEXAGHOST)
        .value("SPHERIC_GUARDIAN", ME::SPHERIC_GUARDIAN)
        .value("CHOSEN", ME::CHOSEN)
        .value("SHELL_PARASITE", ME::SHELL_PARASITE)
        .value("THREE_BYRDS", ME::THREE_BYRDS)
        .value("TWO_THIEVES", ME::TWO_THIEVES)
        .value("CHOSEN_AND_BYRDS", ME::CHOSEN_AND_BYRDS)
        .value("SENTRY_AND_SPHERE", ME::SENTRY_AND_SPHERE)
        .value("SNAKE_PLANT", ME::SNAKE_PLANT)
        .value("SNECKO", ME::SNECKO)
        .value("CENTURION_AND_HEALER", ME::CENTURION_AND_HEALER)
        .value("CULTIST_AND_CHOSEN", ME::CULTIST_AND_CHOSEN)
        .value("THREE_CULTIST", ME::THREE_CULTIST)
        .value("SHELLED_PARASITE_AND_FUNGI", ME::SHELLED_PARASITE_AND_FUNGI)
        .value("GREMLIN_LEADER", ME::GREMLIN_LEADER)
        .value("SLAVERS", ME::SLAVERS)
        .value("BOOK_OF_STABBING", ME::BOOK_OF_STABBING)
        .value("AUTOMATON", ME::AUTOMATON)
        .value("COLLECTOR", ME::COLLECTOR)
        .value("CHAMP", ME::CHAMP)
        .value("THREE_DARKLINGS", ME::THREE_DARKLINGS)
        .value("ORB_WALKER", ME::ORB_WALKER)
        .value("THREE_SHAPES", ME::THREE_SHAPES)
        .value("SPIRE_GROWTH", ME::SPIRE_GROWTH)
        .value("TRANSIENT", ME::TRANSIENT)
        .value("FOUR_SHAPES", ME::FOUR_SHAPES)
        .value("MAW", ME::MAW)
        .value("SPHERE_AND_TWO_SHAPES", ME::SPHERE_AND_TWO_SHAPES)
        .value("JAW_WORM_HORDE", ME::JAW_WORM_HORDE)
        .value("WRITHING_MASS", ME::WRITHING_MASS)
        .value("GIANT_HEAD", ME::GIANT_HEAD)
        .value("NEMESIS", ME::NEMESIS)
        .value("REPTOMANCER", ME::REPTOMANCER)
        .value("AWAKENED_ONE", ME::AWAKENED_ONE)
        .value("TIME_EATER", ME::TIME_EATER)
        .value("DONU_AND_DECA", ME::DONU_AND_DECA)
        .value("SHIELD_AND_SPEAR", ME::SHIELD_AND_SPEAR)
        .value("THE_HEART", ME::THE_HEART)
        .value("LAGAVULIN_EVENT", ME::LAGAVULIN_EVENT)
        .value("COLOSSEUM_EVENT_SLAVERS", ME::COLOSSEUM_EVENT_SLAVERS)
        .value("COLOSSEUM_EVENT_NOBS", ME::COLOSSEUM_EVENT_NOBS)
        .value("MASKED_BANDITS_EVENT", ME::MASKED_BANDITS_EVENT)
        .value("MUSHROOMS_EVENT", ME::MUSHROOMS_EVENT)
        .value("MYSTERIOUS_SPHERE_EVENT", ME::MYSTERIOUS_SPHERE_EVENT);

    pybind11::enum_<RelicId> relicEnum(m, "RelicId");
    relicEnum.value("AKABEKO", RelicId::AKABEKO)
        .value("ART_OF_WAR", RelicId::ART_OF_WAR)
        .value("BIRD_FACED_URN", RelicId::BIRD_FACED_URN)
        .value("BLOODY_IDOL", RelicId::BLOODY_IDOL)
        .value("BLUE_CANDLE", RelicId::BLUE_CANDLE)
        .value("BRIMSTONE", RelicId::BRIMSTONE)
        .value("CALIPERS", RelicId::CALIPERS)
        .value("CAPTAINS_WHEEL", RelicId::CAPTAINS_WHEEL)
        .value("CENTENNIAL_PUZZLE", RelicId::CENTENNIAL_PUZZLE)
        .value("CERAMIC_FISH", RelicId::CERAMIC_FISH)
        .value("CHAMPION_BELT", RelicId::CHAMPION_BELT)
        .value("CHARONS_ASHES", RelicId::CHARONS_ASHES)
        .value("CHEMICAL_X", RelicId::CHEMICAL_X)
        .value("CLOAK_CLASP", RelicId::CLOAK_CLASP)
        .value("DARKSTONE_PERIAPT", RelicId::DARKSTONE_PERIAPT)
        .value("DEAD_BRANCH", RelicId::DEAD_BRANCH)
        .value("DUALITY", RelicId::DUALITY)
        .value("ECTOPLASM", RelicId::ECTOPLASM)
        .value("EMOTION_CHIP", RelicId::EMOTION_CHIP)
        .value("FROZEN_CORE", RelicId::FROZEN_CORE)
        .value("FROZEN_EYE", RelicId::FROZEN_EYE)
        .value("GAMBLING_CHIP", RelicId::GAMBLING_CHIP)
        .value("GINGER", RelicId::GINGER)
        .value("GOLDEN_EYE", RelicId::GOLDEN_EYE)
        .value("GREMLIN_HORN", RelicId::GREMLIN_HORN)
        .value("HAND_DRILL", RelicId::HAND_DRILL)
        .value("HAPPY_FLOWER", RelicId::HAPPY_FLOWER)
        .value("HORN_CLEAT", RelicId::HORN_CLEAT)
        .value("HOVERING_KITE", RelicId::HOVERING_KITE)
        .value("ICE_CREAM", RelicId::ICE_CREAM)
        .value("INCENSE_BURNER", RelicId::INCENSE_BURNER)
        .value("INK_BOTTLE", RelicId::INK_BOTTLE)
        .value("INSERTER", RelicId::INSERTER)
        .value("KUNAI", RelicId::KUNAI)
        .value("LETTER_OPENER", RelicId::LETTER_OPENER)
        .value("LIZARD_TAIL", RelicId::LIZARD_TAIL)
        .value("MAGIC_FLOWER", RelicId::MAGIC_FLOWER)
        .value("MARK_OF_THE_BLOOM", RelicId::MARK_OF_THE_BLOOM)
        .value("MEDICAL_KIT", RelicId::MEDICAL_KIT)
        .value("MELANGE", RelicId::MELANGE)
        .value("MERCURY_HOURGLASS", RelicId::MERCURY_HOURGLASS)
        .value("MUMMIFIED_HAND", RelicId::MUMMIFIED_HAND)
        .value("NECRONOMICON", RelicId::NECRONOMICON)
        .value("NILRYS_CODEX", RelicId::NILRYS_CODEX)
        .value("NUNCHAKU", RelicId::NUNCHAKU)
        .value("ODD_MUSHROOM", RelicId::ODD_MUSHROOM)
        .value("OMAMORI", RelicId::OMAMORI)
        .value("ORANGE_PELLETS", RelicId::ORANGE_PELLETS)
        .value("ORICHALCUM", RelicId::ORICHALCUM)
        .value("ORNAMENTAL_FAN", RelicId::ORNAMENTAL_FAN)
        .value("PAPER_KRANE", RelicId::PAPER_KRANE)
        .value("PAPER_PHROG", RelicId::PAPER_PHROG)
        .value("PEN_NIB", RelicId::PEN_NIB)
        .value("PHILOSOPHERS_STONE", RelicId::PHILOSOPHERS_STONE)
        .value("POCKETWATCH", RelicId::POCKETWATCH)
        .value("RED_SKULL", RelicId::RED_SKULL)
        .value("RUNIC_CUBE", RelicId::RUNIC_CUBE)
        .value("RUNIC_DOME", RelicId::RUNIC_DOME)
        .value("RUNIC_PYRAMID", RelicId::RUNIC_PYRAMID)
        .value("SACRED_BARK", RelicId::SACRED_BARK)
        .value("SELF_FORMING_CLAY", RelicId::SELF_FORMING_CLAY)
        .value("SHURIKEN", RelicId::SHURIKEN)
        .value("SNECKO_EYE", RelicId::SNECKO_EYE)
        .value("SNECKO_SKULL", RelicId::SNECKO_SKULL)
        .value("SOZU", RelicId::SOZU)
        .value("STONE_CALENDAR", RelicId::STONE_CALENDAR)
        .value("STRANGE_SPOON", RelicId::STRANGE_SPOON)
        .value("STRIKE_DUMMY", RelicId::STRIKE_DUMMY)
        .value("SUNDIAL", RelicId::SUNDIAL)
        .value("THE_ABACUS", RelicId::THE_ABACUS)
        .value("THE_BOOT", RelicId::THE_BOOT)
        .value("THE_SPECIMEN", RelicId::THE_SPECIMEN)
        .value("TINGSHA", RelicId::TINGSHA)
        .value("TOOLBOX", RelicId::TOOLBOX)
        .value("TORII", RelicId::TORII)
        .value("TOUGH_BANDAGES", RelicId::TOUGH_BANDAGES)
        .value("TOY_ORNITHOPTER", RelicId::TOY_ORNITHOPTER)
        .value("TUNGSTEN_ROD", RelicId::TUNGSTEN_ROD)
        .value("TURNIP", RelicId::TURNIP)
        .value("TWISTED_FUNNEL", RelicId::TWISTED_FUNNEL)
        .value("UNCEASING_TOP", RelicId::UNCEASING_TOP)
        .value("VELVET_CHOKER", RelicId::VELVET_CHOKER)
        .value("VIOLET_LOTUS", RelicId::VIOLET_LOTUS)
        .value("WARPED_TONGS", RelicId::WARPED_TONGS)
        .value("WRIST_BLADE", RelicId::WRIST_BLADE)
        .value("BLACK_BLOOD", RelicId::BLACK_BLOOD)
        .value("BURNING_BLOOD", RelicId::BURNING_BLOOD)
        .value("MEAT_ON_THE_BONE", RelicId::MEAT_ON_THE_BONE)
        .value("FACE_OF_CLERIC", RelicId::FACE_OF_CLERIC)
        .value("ANCHOR", RelicId::ANCHOR)
        .value("ANCIENT_TEA_SET", RelicId::ANCIENT_TEA_SET)
        .value("BAG_OF_MARBLES", RelicId::BAG_OF_MARBLES)
        .value("BAG_OF_PREPARATION", RelicId::BAG_OF_PREPARATION)
        .value("BLOOD_VIAL", RelicId::BLOOD_VIAL)
        .value("BOTTLED_FLAME", RelicId::BOTTLED_FLAME)
        .value("BOTTLED_LIGHTNING", RelicId::BOTTLED_LIGHTNING)
        .value("BOTTLED_TORNADO", RelicId::BOTTLED_TORNADO)
        .value("BRONZE_SCALES", RelicId::BRONZE_SCALES)
        .value("BUSTED_CROWN", RelicId::BUSTED_CROWN)
        .value("CLOCKWORK_SOUVENIR", RelicId::CLOCKWORK_SOUVENIR)
        .value("COFFEE_DRIPPER", RelicId::COFFEE_DRIPPER)
        .value("CRACKED_CORE", RelicId::CRACKED_CORE)
        .value("CURSED_KEY", RelicId::CURSED_KEY)
        .value("DAMARU", RelicId::DAMARU)
        .value("DATA_DISK", RelicId::DATA_DISK)
        .value("DU_VU_DOLL", RelicId::DU_VU_DOLL)
        .value("ENCHIRIDION", RelicId::ENCHIRIDION)
        .value("FOSSILIZED_HELIX", RelicId::FOSSILIZED_HELIX)
        .value("FUSION_HAMMER", RelicId::FUSION_HAMMER)
        .value("GIRYA", RelicId::GIRYA)
        .value("GOLD_PLATED_CABLES", RelicId::GOLD_PLATED_CABLES)
        .value("GREMLIN_VISAGE", RelicId::GREMLIN_VISAGE)
        .value("HOLY_WATER", RelicId::HOLY_WATER)
        .value("LANTERN", RelicId::LANTERN)
        .value("MARK_OF_PAIN", RelicId::MARK_OF_PAIN)
        .value("MUTAGENIC_STRENGTH", RelicId::MUTAGENIC_STRENGTH)
        .value("NEOWS_LAMENT", RelicId::NEOWS_LAMENT)
        .value("NINJA_SCROLL", RelicId::NINJA_SCROLL)
        .value("NUCLEAR_BATTERY", RelicId::NUCLEAR_BATTERY)
        .value("ODDLY_SMOOTH_STONE", RelicId::ODDLY_SMOOTH_STONE)
        .value("PANTOGRAPH", RelicId::PANTOGRAPH)
        .value("PRESERVED_INSECT", RelicId::PRESERVED_INSECT)
        .value("PURE_WATER", RelicId::PURE_WATER)
        .value("RED_MASK", RelicId::RED_MASK)
        .value("RING_OF_THE_SERPENT", RelicId::RING_OF_THE_SERPENT)
        .value("RING_OF_THE_SNAKE", RelicId::RING_OF_THE_SNAKE)
        .value("RUNIC_CAPACITOR", RelicId::RUNIC_CAPACITOR)
        .value("SLAVERS_COLLAR", RelicId::SLAVERS_COLLAR)
        .value("SLING_OF_COURAGE", RelicId::SLING_OF_COURAGE)
        .value("SYMBIOTIC_VIRUS", RelicId::SYMBIOTIC_VIRUS)
        .value("TEARDROP_LOCKET", RelicId::TEARDROP_LOCKET)
        .value("THREAD_AND_NEEDLE", RelicId::THREAD_AND_NEEDLE)
        .value("VAJRA", RelicId::VAJRA)
        .value("ASTROLABE", RelicId::ASTROLABE)
        .value("BLACK_STAR", RelicId::BLACK_STAR)
        .value("CALLING_BELL", RelicId::CALLING_BELL)
        .value("CAULDRON", RelicId::CAULDRON)
        .value("CULTIST_HEADPIECE", RelicId::CULTIST_HEADPIECE)
        .value("DOLLYS_MIRROR", RelicId::DOLLYS_MIRROR)
        .value("DREAM_CATCHER", RelicId::DREAM_CATCHER)
        .value("EMPTY_CAGE", RelicId::EMPTY_CAGE)
        .value("ETERNAL_FEATHER", RelicId::ETERNAL_FEATHER)
        .value("FROZEN_EGG", RelicId::FROZEN_EGG)
        .value("GOLDEN_IDOL", RelicId::GOLDEN_IDOL)
        .value("JUZU_BRACELET", RelicId::JUZU_BRACELET)
        .value("LEES_WAFFLE", RelicId::LEES_WAFFLE)
        .value("MANGO", RelicId::MANGO)
        .value("MATRYOSHKA", RelicId::MATRYOSHKA)
        .value("MAW_BANK", RelicId::MAW_BANK)
        .value("MEAL_TICKET", RelicId::MEAL_TICKET)
        .value("MEMBERSHIP_CARD", RelicId::MEMBERSHIP_CARD)
        .value("MOLTEN_EGG", RelicId::MOLTEN_EGG)
        .value("NLOTHS_GIFT", RelicId::NLOTHS_GIFT)
        .value("NLOTHS_HUNGRY_FACE", RelicId::NLOTHS_HUNGRY_FACE)
        .value("OLD_COIN", RelicId::OLD_COIN)
        .value("ORRERY", RelicId::ORRERY)
        .value("PANDORAS_BOX", RelicId::PANDORAS_BOX)
        .value("PEACE_PIPE", RelicId::PEACE_PIPE)
        .value("PEAR", RelicId::PEAR)
        .value("POTION_BELT", RelicId::POTION_BELT)
        .value("PRAYER_WHEEL", RelicId::PRAYER_WHEEL)
        .value("PRISMATIC_SHARD", RelicId::PRISMATIC_SHARD)
        .value("QUESTION_CARD", RelicId::QUESTION_CARD)
        .value("REGAL_PILLOW", RelicId::REGAL_PILLOW)
        .value("SSSERPENT_HEAD", RelicId::SSSERPENT_HEAD)
        .value("SHOVEL", RelicId::SHOVEL)
        .value("SINGING_BOWL", RelicId::SINGING_BOWL)
        .value("SMILING_MASK", RelicId::SMILING_MASK)
        .value("SPIRIT_POOP", RelicId::SPIRIT_POOP)
        .value("STRAWBERRY", RelicId::STRAWBERRY)
        .value("THE_COURIER", RelicId::THE_COURIER)
        .value("TINY_CHEST", RelicId::TINY_CHEST)
        .value("TINY_HOUSE", RelicId::TINY_HOUSE)
        .value("TOXIC_EGG", RelicId::TOXIC_EGG)
        .value("WAR_PAINT", RelicId::WAR_PAINT)
        .value("WHETSTONE", RelicId::WHETSTONE)
        .value("WHITE_BEAST_STATUE", RelicId::WHITE_BEAST_STATUE)
        .value("WING_BOOTS", RelicId::WING_BOOTS)
        .value("CIRCLET", RelicId::CIRCLET)
        .value("RED_CIRCLET", RelicId::RED_CIRCLET)
        .value("INVALID", RelicId::INVALID);

    pybind11::enum_<MonsterId> monsterEnum(m, "MonsterId");
    monsterEnum.value("INVALID", MonsterId::INVALID)
        .value("ACID_SLIME_L", MonsterId::ACID_SLIME_L)
        .value("ACID_SLIME_M", MonsterId::ACID_SLIME_M)
        .value("ACID_SLIME_S", MonsterId::ACID_SLIME_S)
.value("        AWAKENED_ONE", MonsterId::        AWAKENED_ONE)
        .value("BEAR", MonsterId::BEAR)
        .value("BLUE_SLAVER", MonsterId::BLUE_SLAVER)
        .value("BOOK_OF_STABBING", MonsterId::BOOK_OF_STABBING)
        .value("BRONZE_AUTOMATON", MonsterId::BRONZE_AUTOMATON)
        .value("BRONZE_ORB", MonsterId::BRONZE_ORB)
        .value("BYRD", MonsterId::BYRD)
        .value("CENTURION", MonsterId::CENTURION)
        .value("CHOSEN", MonsterId::CHOSEN)
        .value("CORRUPT_HEART", MonsterId::CORRUPT_HEART)
        .value("CULTIST", MonsterId::CULTIST)
        .value("DAGGER", MonsterId::DAGGER)
        .value("DARKLING", MonsterId::DARKLING)
        .value("DECA", MonsterId::DECA)
        .value("DONU", MonsterId::DONU)
        .value("EXPLODER", MonsterId::EXPLODER)
        .value("FAT_GREMLIN", MonsterId::FAT_GREMLIN)
        .value("FUNGI_BEAST", MonsterId::FUNGI_BEAST)
        .value("GIANT_HEAD", MonsterId::GIANT_HEAD)
        .value("GREEN_LOUSE", MonsterId::GREEN_LOUSE)
        .value("GREMLIN_LEADER", MonsterId::GREMLIN_LEADER)
        .value("GREMLIN_NOB", MonsterId::GREMLIN_NOB)
        .value("GREMLIN_WIZARD", MonsterId::GREMLIN_WIZARD)
        .value("HEXAGHOST", MonsterId::HEXAGHOST)
        .value("JAW_WORM", MonsterId::JAW_WORM)
        .value("LAGAVULIN", MonsterId::LAGAVULIN)
        .value("LOOTER", MonsterId::LOOTER)
        .value("MAD_GREMLIN", MonsterId::MAD_GREMLIN)
        .value("MUGGER", MonsterId::MUGGER)
        .value("MYSTIC", MonsterId::MYSTIC)
        .value("NEMESIS", MonsterId::NEMESIS)
        .value("ORB_WALKER", MonsterId::ORB_WALKER)
        .value("POINTY", MonsterId::POINTY)
        .value("RED_LOUSE", MonsterId::RED_LOUSE)
        .value("RED_SLAVER", MonsterId::RED_SLAVER)
        .value("REPTOMANCER", MonsterId::REPTOMANCER)
        .value("REPULSOR", MonsterId::REPULSOR)
        .value("ROMEO", MonsterId::ROMEO)
        .value("SENTRY", MonsterId::SENTRY)
        .value("SHELLED_PARASITE", MonsterId::SHELLED_PARASITE)
        .value("SHIELD_GREMLIN", MonsterId::SHIELD_GREMLIN)
        .value("SLIME_BOSS", MonsterId::SLIME_BOSS)
        .value("SNAKE_PLANT", MonsterId::SNAKE_PLANT)
        .value("SNEAKY_GREMLIN", MonsterId::SNEAKY_GREMLIN)
        .value("SNECKO", MonsterId::SNECKO)
        .value("SPHERIC_GUARDIAN", MonsterId::SPHERIC_GUARDIAN)
        .value("SPIKER", MonsterId::SPIKER)
        .value("SPIKE_SLIME_L", MonsterId::SPIKE_SLIME_L)
        .value("SPIKE_SLIME_M", MonsterId::SPIKE_SLIME_M)
        .value("SPIKE_SLIME_S", MonsterId::SPIKE_SLIME_S)
        .value("SPIRE_GROWTH", MonsterId::SPIRE_GROWTH)
        .value("SPIRE_SHIELD", MonsterId::SPIRE_SHIELD)
        .value("SPIRE_SPEAR", MonsterId::SPIRE_SPEAR)
        .value("TASKMASTER", MonsterId::TASKMASTER)
        .value("THE_CHAMP", MonsterId::THE_CHAMP)
        .value("THE_COLLECTOR", MonsterId::THE_COLLECTOR)
        .value("THE_GUARDIAN", MonsterId::THE_GUARDIAN)
        .value("THE_MAW", MonsterId::THE_MAW)
        .value("TIME_EATER", MonsterId::TIME_EATER)
        .value("TORCH_HEAD", MonsterId::TORCH_HEAD)
        .value("TRANSIENT", MonsterId::TRANSIENT)
        .value("WRITHING_MASS", MonsterId::WRITHING_MASS);


    pybind11::enum_<MonsterMoveId> monsterMoveEnum(m, "MonsterMoveId");
    monsterMoveEnum.value("INVALID", MonsterMoveId::INVALID)
        .value("GENERIC_ESCAPE_MOVE", MonsterMoveId::GENERIC_ESCAPE_MOVE)
        .value("ACID_SLIME_L_CORROSIVE_SPIT", MonsterMoveId::ACID_SLIME_L_CORROSIVE_SPIT)
        .value("ACID_SLIME_L_LICK", MonsterMoveId::ACID_SLIME_L_LICK)
        .value("ACID_SLIME_L_TACKLE", MonsterMoveId::ACID_SLIME_L_TACKLE)
        .value("ACID_SLIME_L_SPLIT", MonsterMoveId::ACID_SLIME_L_SPLIT)
        .value("ACID_SLIME_M_CORROSIVE_SPIT", MonsterMoveId::ACID_SLIME_M_CORROSIVE_SPIT)
        .value("ACID_SLIME_M_LICK", MonsterMoveId::ACID_SLIME_M_LICK)
        .value("ACID_SLIME_M_TACKLE", MonsterMoveId::ACID_SLIME_M_TACKLE)
        .value("ACID_SLIME_S_LICK", MonsterMoveId::ACID_SLIME_S_LICK)
        .value("ACID_SLIME_S_TACKLE", MonsterMoveId::ACID_SLIME_S_TACKLE)
        .value("AWAKENED_ONE_SLASH", MonsterMoveId::AWAKENED_ONE_SLASH)
        .value("AWAKENED_ONE_SOUL_STRIKE", MonsterMoveId::AWAKENED_ONE_SOUL_STRIKE)
        .value("AWAKENED_ONE_REBIRTH", MonsterMoveId::AWAKENED_ONE_REBIRTH)
        .value("AWAKENED_ONE_DARK_ECHO", MonsterMoveId::AWAKENED_ONE_DARK_ECHO)
        .value("AWAKENED_ONE_SLUDGE", MonsterMoveId::AWAKENED_ONE_SLUDGE)
        .value("AWAKENED_ONE_TACKLE", MonsterMoveId::AWAKENED_ONE_TACKLE)
        .value("BEAR_BEAR_HUG", MonsterMoveId::BEAR_BEAR_HUG)
        .value("BEAR_LUNGE", MonsterMoveId::BEAR_LUNGE)
        .value("BEAR_MAUL", MonsterMoveId::BEAR_MAUL)
        .value("BLUE_SLAVER_STAB", MonsterMoveId::BLUE_SLAVER_STAB)
        .value("BLUE_SLAVER_RAKE", MonsterMoveId::BLUE_SLAVER_RAKE)
        .value("BOOK_OF_STABBING_MULTI_STAB", MonsterMoveId::BOOK_OF_STABBING_MULTI_STAB)
        .value("BOOK_OF_STABBING_SINGLE_STAB", MonsterMoveId::BOOK_OF_STABBING_SINGLE_STAB)
        .value("BRONZE_AUTOMATON_BOOST", MonsterMoveId::BRONZE_AUTOMATON_BOOST)
        .value("BRONZE_AUTOMATON_FLAIL", MonsterMoveId::BRONZE_AUTOMATON_FLAIL)
        .value("BRONZE_AUTOMATON_HYPER_BEAM", MonsterMoveId::BRONZE_AUTOMATON_HYPER_BEAM)
        .value("BRONZE_AUTOMATON_SPAWN_ORBS", MonsterMoveId::BRONZE_AUTOMATON_SPAWN_ORBS)
        .value("BRONZE_AUTOMATON_STUNNED", MonsterMoveId::BRONZE_AUTOMATON_STUNNED)
        .value("BRONZE_ORB_BEAM", MonsterMoveId::BRONZE_ORB_BEAM)
        .value("BRONZE_ORB_STASIS", MonsterMoveId::BRONZE_ORB_STASIS)
        .value("BRONZE_ORB_SUPPORT_BEAM", MonsterMoveId::BRONZE_ORB_SUPPORT_BEAM)
        .value("BYRD_CAW", MonsterMoveId::BYRD_CAW)
        .value("BYRD_FLY", MonsterMoveId::BYRD_FLY)
        .value("BYRD_HEADBUTT", MonsterMoveId::BYRD_HEADBUTT)
        .value("BYRD_PECK", MonsterMoveId::BYRD_PECK)
        .value("BYRD_STUNNED", MonsterMoveId::BYRD_STUNNED)
        .value("BYRD_SWOOP", MonsterMoveId::BYRD_SWOOP)
        .value("CENTURION_SLASH", MonsterMoveId::CENTURION_SLASH)
        .value("CENTURION_FURY", MonsterMoveId::CENTURION_FURY)
        .value("CENTURION_DEFEND", MonsterMoveId::CENTURION_DEFEND)
        .value("CHOSEN_POKE", MonsterMoveId::CHOSEN_POKE)
        .value("CHOSEN_ZAP", MonsterMoveId::CHOSEN_ZAP)
        .value("CHOSEN_DEBILITATE", MonsterMoveId::CHOSEN_DEBILITATE)
        .value("CHOSEN_DRAIN", MonsterMoveId::CHOSEN_DRAIN)
        .value("CHOSEN_HEX", MonsterMoveId::CHOSEN_HEX)
        .value("CORRUPT_HEART_DEBILITATE", MonsterMoveId::CORRUPT_HEART_DEBILITATE)
        .value("CORRUPT_HEART_BLOOD_SHOTS", MonsterMoveId::CORRUPT_HEART_BLOOD_SHOTS)
        .value("CORRUPT_HEART_ECHO", MonsterMoveId::CORRUPT_HEART_ECHO)
        .value("CORRUPT_HEART_BUFF", MonsterMoveId::CORRUPT_HEART_BUFF)
        .value("CULTIST_INCANTATION", MonsterMoveId::CULTIST_INCANTATION)
        .value("CULTIST_DARK_STRIKE", MonsterMoveId::CULTIST_DARK_STRIKE)
        .value("DAGGER_STAB", MonsterMoveId::DAGGER_STAB)
        .value("DAGGER_EXPLODE", MonsterMoveId::DAGGER_EXPLODE)
        .value("DARKLING_NIP", MonsterMoveId::DARKLING_NIP)
        .value("DARKLING_CHOMP", MonsterMoveId::DARKLING_CHOMP)
        .value("DARKLING_HARDEN", MonsterMoveId::DARKLING_HARDEN)
        .value("DARKLING_REINCARNATE", MonsterMoveId::DARKLING_REINCARNATE)
        .value("DARKLING_REGROW", MonsterMoveId::DARKLING_REGROW)
        .value("DECA_SQUARE_OF_PROTECTION", MonsterMoveId::DECA_SQUARE_OF_PROTECTION)
        .value("DECA_BEAM", MonsterMoveId::DECA_BEAM)
        .value("DONU_CIRCLE_OF_POWER", MonsterMoveId::DONU_CIRCLE_OF_POWER)
        .value("DONU_BEAM", MonsterMoveId::DONU_BEAM)
        .value("EXPLODER_SLAM", MonsterMoveId::EXPLODER_SLAM)
        .value("EXPLODER_EXPLODE", MonsterMoveId::EXPLODER_EXPLODE)
        .value("FAT_GREMLIN_SMASH", MonsterMoveId::FAT_GREMLIN_SMASH)
        .value("FUNGI_BEAST_BITE", MonsterMoveId::FUNGI_BEAST_BITE)
        .value("FUNGI_BEAST_GROW", MonsterMoveId::FUNGI_BEAST_GROW)
        .value("GIANT_HEAD_COUNT", MonsterMoveId::GIANT_HEAD_COUNT)
        .value("GIANT_HEAD_GLARE", MonsterMoveId::GIANT_HEAD_GLARE)
        .value("GIANT_HEAD_IT_IS_TIME", MonsterMoveId::GIANT_HEAD_IT_IS_TIME)
        .value("GREEN_LOUSE_BITE", MonsterMoveId::GREEN_LOUSE_BITE)
        .value("GREEN_LOUSE_SPIT_WEB", MonsterMoveId::GREEN_LOUSE_SPIT_WEB)
        .value("GREMLIN_LEADER_ENCOURAGE", MonsterMoveId::GREMLIN_LEADER_ENCOURAGE)
        .value("GREMLIN_LEADER_RALLY", MonsterMoveId::GREMLIN_LEADER_RALLY)
        .value("GREMLIN_LEADER_STAB", MonsterMoveId::GREMLIN_LEADER_STAB)
        .value("GREMLIN_NOB_BELLOW", MonsterMoveId::GREMLIN_NOB_BELLOW)
        .value("GREMLIN_NOB_RUSH", MonsterMoveId::GREMLIN_NOB_RUSH)
        .value("GREMLIN_NOB_SKULL_BASH", MonsterMoveId::GREMLIN_NOB_SKULL_BASH)
        .value("GREMLIN_WIZARD_CHARGING", MonsterMoveId::GREMLIN_WIZARD_CHARGING)
        .value("GREMLIN_WIZARD_ULTIMATE_BLAST", MonsterMoveId::GREMLIN_WIZARD_ULTIMATE_BLAST)
        .value("HEXAGHOST_ACTIVATE", MonsterMoveId::HEXAGHOST_ACTIVATE)
        .value("HEXAGHOST_DIVIDER", MonsterMoveId::HEXAGHOST_DIVIDER)
        .value("HEXAGHOST_INFERNO", MonsterMoveId::HEXAGHOST_INFERNO)
        .value("HEXAGHOST_SEAR", MonsterMoveId::HEXAGHOST_SEAR)
        .value("HEXAGHOST_TACKLE", MonsterMoveId::HEXAGHOST_TACKLE)
        .value("HEXAGHOST_INFLAME", MonsterMoveId::HEXAGHOST_INFLAME)
        .value("JAW_WORM_CHOMP", MonsterMoveId::JAW_WORM_CHOMP)
        .value("JAW_WORM_THRASH", MonsterMoveId::JAW_WORM_THRASH)
        .value("JAW_WORM_BELLOW", MonsterMoveId::JAW_WORM_BELLOW)
        .value("LAGAVULIN_ATTACK", MonsterMoveId::LAGAVULIN_ATTACK)
        .value("LAGAVULIN_SIPHON_SOUL", MonsterMoveId::LAGAVULIN_SIPHON_SOUL)
        .value("LAGAVULIN_SLEEP", MonsterMoveId::LAGAVULIN_SLEEP)
        .value("LOOTER_MUG", MonsterMoveId::LOOTER_MUG)
        .value("LOOTER_LUNGE", MonsterMoveId::LOOTER_LUNGE)
        .value("LOOTER_SMOKE_BOMB", MonsterMoveId::LOOTER_SMOKE_BOMB)
        .value("LOOTER_ESCAPE", MonsterMoveId::LOOTER_ESCAPE)
        .value("MAD_GREMLIN_SCRATCH", MonsterMoveId::MAD_GREMLIN_SCRATCH)
        .value("MUGGER_MUG", MonsterMoveId::MUGGER_MUG)
        .value("MUGGER_LUNGE", MonsterMoveId::MUGGER_LUNGE)
        .value("MUGGER_SMOKE_BOMB", MonsterMoveId::MUGGER_SMOKE_BOMB)
        .value("MUGGER_ESCAPE", MonsterMoveId::MUGGER_ESCAPE)
        .value("MYSTIC_HEAL", MonsterMoveId::MYSTIC_HEAL)
        .value("MYSTIC_BUFF", MonsterMoveId::MYSTIC_BUFF)
        .value("MYSTIC_ATTACK_DEBUFF", MonsterMoveId::MYSTIC_ATTACK_DEBUFF)
        .value("NEMESIS_DEBUFF", MonsterMoveId::NEMESIS_DEBUFF)
        .value("NEMESIS_ATTACK", MonsterMoveId::NEMESIS_ATTACK)
        .value("NEMESIS_SCYTHE", MonsterMoveId::NEMESIS_SCYTHE)
        .value("ORB_WALKER_LASER", MonsterMoveId::ORB_WALKER_LASER)
        .value("ORB_WALKER_CLAW", MonsterMoveId::ORB_WALKER_CLAW)
        .value("POINTY_ATTACK", MonsterMoveId::POINTY_ATTACK)
        .value("RED_LOUSE_BITE", MonsterMoveId::RED_LOUSE_BITE)
        .value("RED_LOUSE_GROW", MonsterMoveId::RED_LOUSE_GROW)
        .value("RED_SLAVER_STAB", MonsterMoveId::RED_SLAVER_STAB)
        .value("RED_SLAVER_SCRAPE", MonsterMoveId::RED_SLAVER_SCRAPE)
        .value("RED_SLAVER_ENTANGLE", MonsterMoveId::RED_SLAVER_ENTANGLE)
        .value("REPTOMANCER_SUMMON", MonsterMoveId::REPTOMANCER_SUMMON)
        .value("REPTOMANCER_SNAKE_STRIKE", MonsterMoveId::REPTOMANCER_SNAKE_STRIKE)
        .value("REPTOMANCER_BIG_BITE", MonsterMoveId::REPTOMANCER_BIG_BITE)
        .value("REPULSOR_BASH", MonsterMoveId::REPULSOR_BASH)
        .value("REPULSOR_REPULSE", MonsterMoveId::REPULSOR_REPULSE)
        .value("ROMEO_MOCK", MonsterMoveId::ROMEO_MOCK)
        .value("ROMEO_AGONIZING_SLASH", MonsterMoveId::ROMEO_AGONIZING_SLASH)
        .value("ROMEO_CROSS_SLASH", MonsterMoveId::ROMEO_CROSS_SLASH)
        .value("SENTRY_BEAM", MonsterMoveId::SENTRY_BEAM)
        .value("SENTRY_BOLT", MonsterMoveId::SENTRY_BOLT)
        .value("SHELLED_PARASITE_DOUBLE_STRIKE", MonsterMoveId::SHELLED_PARASITE_DOUBLE_STRIKE)
        .value("SHELLED_PARASITE_FELL", MonsterMoveId::SHELLED_PARASITE_FELL)
        .value("SHELLED_PARASITE_STUNNED", MonsterMoveId::SHELLED_PARASITE_STUNNED)
        .value("SHELLED_PARASITE_SUCK", MonsterMoveId::SHELLED_PARASITE_SUCK)
        .value("SHIELD_GREMLIN_PROTECT", MonsterMoveId::SHIELD_GREMLIN_PROTECT)
        .value("SHIELD_GREMLIN_SHIELD_BASH", MonsterMoveId::SHIELD_GREMLIN_SHIELD_BASH)
        .value("SLIME_BOSS_GOOP_SPRAY", MonsterMoveId::SLIME_BOSS_GOOP_SPRAY)
        .value("SLIME_BOSS_PREPARING", MonsterMoveId::SLIME_BOSS_PREPARING)
        .value("SLIME_BOSS_SLAM", MonsterMoveId::SLIME_BOSS_SLAM)
        .value("SLIME_BOSS_SPLIT", MonsterMoveId::SLIME_BOSS_SPLIT)
        .value("SNAKE_PLANT_CHOMP", MonsterMoveId::SNAKE_PLANT_CHOMP)
        .value("SNAKE_PLANT_ENFEEBLING_SPORES", MonsterMoveId::SNAKE_PLANT_ENFEEBLING_SPORES)
        .value("SNEAKY_GREMLIN_PUNCTURE", MonsterMoveId::SNEAKY_GREMLIN_PUNCTURE)
        .value("SNECKO_PERPLEXING_GLARE", MonsterMoveId::SNECKO_PERPLEXING_GLARE)
        .value("SNECKO_TAIL_WHIP", MonsterMoveId::SNECKO_TAIL_WHIP)
        .value("SNECKO_BITE", MonsterMoveId::SNECKO_BITE)
        .value("SPHERIC_GUARDIAN_SLAM", MonsterMoveId::SPHERIC_GUARDIAN_SLAM)
        .value("SPHERIC_GUARDIAN_ACTIVATE", MonsterMoveId::SPHERIC_GUARDIAN_ACTIVATE)
        .value("SPHERIC_GUARDIAN_HARDEN", MonsterMoveId::SPHERIC_GUARDIAN_HARDEN)
        .value("SPHERIC_GUARDIAN_ATTACK_DEBUFF", MonsterMoveId::SPHERIC_GUARDIAN_ATTACK_DEBUFF)
        .value("SPIKER_CUT", MonsterMoveId::SPIKER_CUT)
        .value("SPIKER_SPIKE", MonsterMoveId::SPIKER_SPIKE)
        .value("SPIKE_SLIME_L_FLAME_TACKLE", MonsterMoveId::SPIKE_SLIME_L_FLAME_TACKLE)
        .value("SPIKE_SLIME_L_LICK", MonsterMoveId::SPIKE_SLIME_L_LICK)
        .value("SPIKE_SLIME_L_SPLIT", MonsterMoveId::SPIKE_SLIME_L_SPLIT)
        .value("SPIKE_SLIME_M_FLAME_TACKLE", MonsterMoveId::SPIKE_SLIME_M_FLAME_TACKLE)
        .value("SPIKE_SLIME_M_LICK", MonsterMoveId::SPIKE_SLIME_M_LICK)
        .value("SPIKE_SLIME_S_TACKLE", MonsterMoveId::SPIKE_SLIME_S_TACKLE)
        .value("SPIRE_GROWTH_QUICK_TACKLE", MonsterMoveId::SPIRE_GROWTH_QUICK_TACKLE)
        .value("SPIRE_GROWTH_SMASH", MonsterMoveId::SPIRE_GROWTH_SMASH)
        .value("SPIRE_GROWTH_CONSTRICT", MonsterMoveId::SPIRE_GROWTH_CONSTRICT)
        .value("SPIRE_SHIELD_BASH", MonsterMoveId::SPIRE_SHIELD_BASH)
        .value("SPIRE_SHIELD_FORTIFY", MonsterMoveId::SPIRE_SHIELD_FORTIFY)
        .value("SPIRE_SHIELD_SMASH", MonsterMoveId::SPIRE_SHIELD_SMASH)
        .value("SPIRE_SPEAR_BURN_STRIKE", MonsterMoveId::SPIRE_SPEAR_BURN_STRIKE)
        .value("SPIRE_SPEAR_PIERCER", MonsterMoveId::SPIRE_SPEAR_PIERCER)
        .value("SPIRE_SPEAR_SKEWER", MonsterMoveId::SPIRE_SPEAR_SKEWER)
        .value("TASKMASTER_SCOURING_WHIP", MonsterMoveId::TASKMASTER_SCOURING_WHIP)
        .value("TORCH_HEAD_TACKLE", MonsterMoveId::TORCH_HEAD_TACKLE)
        .value("THE_CHAMP_DEFENSIVE_STANCE", MonsterMoveId::THE_CHAMP_DEFENSIVE_STANCE)
        .value("THE_CHAMP_FACE_SLAP", MonsterMoveId::THE_CHAMP_FACE_SLAP)
        .value("THE_CHAMP_TAUNT", MonsterMoveId::THE_CHAMP_TAUNT)
        .value("THE_CHAMP_HEAVY_SLASH", MonsterMoveId::THE_CHAMP_HEAVY_SLASH)
        .value("THE_CHAMP_GLOAT", MonsterMoveId::THE_CHAMP_GLOAT)
        .value("THE_CHAMP_EXECUTE", MonsterMoveId::THE_CHAMP_EXECUTE)
        .value("THE_CHAMP_ANGER", MonsterMoveId::THE_CHAMP_ANGER)
        .value("THE_COLLECTOR_BUFF", MonsterMoveId::THE_COLLECTOR_BUFF)
        .value("THE_COLLECTOR_FIREBALL", MonsterMoveId::THE_COLLECTOR_FIREBALL)
        .value("THE_COLLECTOR_MEGA_DEBUFF", MonsterMoveId::THE_COLLECTOR_MEGA_DEBUFF)
        .value("THE_COLLECTOR_SPAWN", MonsterMoveId::THE_COLLECTOR_SPAWN)
        .value("THE_GUARDIAN_CHARGING_UP", MonsterMoveId::THE_GUARDIAN_CHARGING_UP)
        .value("THE_GUARDIAN_FIERCE_BASH", MonsterMoveId::THE_GUARDIAN_FIERCE_BASH)
        .value("THE_GUARDIAN_VENT_STEAM", MonsterMoveId::THE_GUARDIAN_VENT_STEAM)
        .value("THE_GUARDIAN_WHIRLWIND", MonsterMoveId::THE_GUARDIAN_WHIRLWIND)
        .value("THE_GUARDIAN_DEFENSIVE_MODE", MonsterMoveId::THE_GUARDIAN_DEFENSIVE_MODE)
        .value("THE_GUARDIAN_ROLL_ATTACK", MonsterMoveId::THE_GUARDIAN_ROLL_ATTACK)
        .value("THE_GUARDIAN_TWIN_SLAM", MonsterMoveId::THE_GUARDIAN_TWIN_SLAM)
        .value("THE_MAW_ROAR", MonsterMoveId::THE_MAW_ROAR)
        .value("THE_MAW_DROOL", MonsterMoveId::THE_MAW_DROOL)
        .value("THE_MAW_SLAM", MonsterMoveId::THE_MAW_SLAM)
        .value("THE_MAW_NOM", MonsterMoveId::THE_MAW_NOM)
        .value("TIME_EATER_REVERBERATE", MonsterMoveId::TIME_EATER_REVERBERATE)
        .value("TIME_EATER_HEAD_SLAM", MonsterMoveId::TIME_EATER_HEAD_SLAM)
        .value("TIME_EATER_RIPPLE", MonsterMoveId::TIME_EATER_RIPPLE)
        .value("TIME_EATER_HASTE", MonsterMoveId::TIME_EATER_HASTE)
        .value("TRANSIENT_ATTACK", MonsterMoveId::TRANSIENT_ATTACK)
        .value("WRITHING_MASS_IMPLANT", MonsterMoveId::WRITHING_MASS_IMPLANT)
        .value("WRITHING_MASS_FLAIL", MonsterMoveId::WRITHING_MASS_FLAIL)
        .value("WRITHING_MASS_WITHER", MonsterMoveId::WRITHING_MASS_WITHER)
        .value("WRITHING_MASS_MULTI_STRIKE", MonsterMoveId::WRITHING_MASS_MULTI_STRIKE)
        .value("WRITHING_MASS_STRONG_STRIKE", MonsterMoveId::WRITHING_MASS_STRONG_STRIKE);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

// os.add_dll_directory("C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin")



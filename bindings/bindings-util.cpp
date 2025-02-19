//
// Created by keega on 9/24/2021.
//
#include <sstream>
#include <algorithm>

#include "combat/BattleContext.h"
#include "combat/Player.h"
#include "constants/PlayerStatusEffects.h"
#include "sim/ConsoleSimulator.h"
#include "sim/search/ScumSearchAgent2.h"
#include "sim/SimHelpers.h"
#include "sim/PrintHelpers.h"
#include "game/Game.h"
#include "game/Map.h"

#include "slaythespire.h"

namespace sts {

    std::array<float, 22> RLInterface::getMonsterEmbedding(Monster &monster) {
        std::array<float, 22> ret{};
        ret[0] = (float) monster.id;
        ret[1] = monster.curHp;
        ret[2] = monster.maxHp;
        ret[3] = monster.block;
        ret[4] = monster.statusBits;
        ret[5] = monster.artifact;
        ret[6] = monster.blockReturn;
        ret[7] = monster.choked;
        ret[8] = monster.corpseExplosion;
        ret[9] = monster.lockOn;
        ret[10] = monster.mark;
        ret[11] = monster.metallicize;
        ret[12] = monster.platedArmor;
        ret[13] = monster.poison;
        ret[14] = monster.regen;
        ret[15] = monster.shackled;
        ret[16] = monster.strength;
        ret[17] = monster.vulnerable;
        ret[18] = monster.weak;
        ret[19] = monster.uniquePower0;
        ret[20] = monster.uniquePower1;
        ret[21] = (float)monster.moveHistory[0];
        return ret;
    }


    std::array<float, 100> RLInterface::getPlayerEmbedding(Player &player) {
        std::array<float, 100> ret{};
        ret[0] = (float) player.cc;
        ret[1] = player.curHp;
        ret[2] = player.maxHp;
        ret[3] = player.energy;
        ret[4] = player.energyPerTurn;
        ret[5] = player.cardDrawPerTurn;
        ret[6] = (float) player.stance;
        ret[7] = player.orbSlots;
        ret[8] = player.block;
        ret[9] = player.artifact;
        ret[10] = player.dexterity;
        ret[11] = player.focus;
        ret[12] = player.strength;

        for(int i = 0; i < 87; i++) {
            ret[13+i] = player.getStatusRuntime(static_cast<PlayerStatus>(i));
        }
        return ret;
    }

    void RLInterface::prettyPrintStateEmbedding(GameContext &gc, BattleContext &bc) {
        std::cout << "---------------------------------------" << std::endl;

        Player player = bc.player;
        std::cout << "player.cc: " << (int)player.cc << std::endl;
        std::cout << "player.curHp: " << player.curHp << std::endl;
        std::cout << "player.maxHp: " << player.maxHp << std::endl;
        std::cout << "player.energy: " << player.energy << std::endl;
        std::cout << "player.energyPerTurn: " << player.energyPerTurn << std::endl;
        std::cout << "player.cardDrawPerTurn: " << player.cardDrawPerTurn << std::endl;
        std::cout << "player.stance: " << (int)player.stance << std::endl;
        std::cout << "player.orbSlots: " << player.orbSlots << std::endl;
        std::cout << "player.block: " << player.block << std::endl;
        std::cout << "player.artifact: " << player.artifact << std::endl;
        std::cout << "player.dexterity: " << player.dexterity << std::endl;
        std::cout << "player.focus: " << player.focus << std::endl;
        std::cout << "player.strength: " << player.strength << std::endl;
        for(int i = 0; i < 87; i++) {
            PlayerStatus status = static_cast<PlayerStatus>(i);
            int statusState = player.getStatusRuntime(status);
            if (statusState > 0)
                std::cout << "player.status. " << playerStatusStrings[i] << ": " << statusState << std::endl;
        }

        std::cout << "---------------------------------------" << std::endl;

        for (int i = 0; i < 5; i++) {
            Monster monster = bc.monsters.arr[i];
            if (monster.id == MonsterId::INVALID)
                continue;
            std::cout << "monster number " << i << std::endl;
            std::cout << "monster.id: " << (int) monster.id << std::endl;
            std::cout << "monster.curHp: " << monster.curHp << std::endl;
            std::cout << "monster.maxHp: " << monster.maxHp << std::endl;
            std::cout << "monster.block: " << monster.block << std::endl;
            std::cout << "monster.statusBits: " << monster.statusBits << std::endl;
            std::cout << "monster.artifact: " << monster.artifact << std::endl;
            std::cout << "monster.blockReturn: " << monster.blockReturn << std::endl;
            std::cout << "monster.choked: " << monster.choked << std::endl;
            std::cout << "monster.corpseExplosion: " << monster.corpseExplosion << std::endl;
            std::cout << "monster.lockOn: " << monster.lockOn << std::endl;
            std::cout << "monster.mark: " << monster.mark << std::endl;
            std::cout << "monster.metallicize: " << monster.metallicize << std::endl;
            std::cout << "monster.platedArmor: " << monster.platedArmor << std::endl;
            std::cout << "monster.poison: " << monster.poison << std::endl;
            std::cout << "monster.regen: " << monster.regen << std::endl;
            std::cout << "monster.shackled: " << monster.shackled << std::endl;
            std::cout << "monster.strength: " << monster.strength << std::endl;
            std::cout << "monster.vulnerable: " << monster.vulnerable << std::endl;
            std::cout << "monster.weak: " << monster.weak << std::endl;
            std::cout << "monster.uniquePower0: " << monster.uniquePower0 << std::endl;
            std::cout << "monster.uniquePower1: " << monster.uniquePower1 << std::endl;
            std::cout << "monster.nextMove: " << (float)monster.moveHistory[0] << std::endl;
            std::cout << "---------------------------------------" << std::endl;
        }
        std::cout << "cardsInHand: " << bc.cards.cardsInHand << std::endl;

        std::cout << "---------------------------------------" << std::endl;
    }

    std::array<float, 214> RLInterface::getStateEmbedding(GameContext &gc, BattleContext &bc) {
        std::array<float, 214> ret{};

        for(int i = 0; i < 5; i++) {
            std::array<float, 22> monsterEmbedding = RLInterface::getMonsterEmbedding(bc.monsters.arr[0]);
            std::copy_n(monsterEmbedding.begin(), 22, ret.begin()+4+22*i);
        }

        std::array<float, 100> playerEmbedding = RLInterface::getPlayerEmbedding(bc.player);
        std::copy_n(playerEmbedding.begin(), 100, ret.begin()+114);

        ret[213] = bc.cards.cardsInHand;

        return ret;
    }

    NNInterface::NNInterface() :
            cardEncodeMap(createOneHotCardEncodingMap()),
            bossEncodeMap(createBossEncodingMap()) {}

    int NNInterface::getCardIdx(Card c) const {
        int idx = cardEncodeMap[static_cast<int>(c.id)] * 2;
        if (idx == -1) {
            std::cerr << "attemped to get encoding idx for invalid card" << std::endl;
            assert(false);
        }

        if (c.isUpgraded()) {
            idx += 1;
        }

        return idx;
    }

    std::array<int,NNInterface::observation_space_size> NNInterface::getObservation(const GameContext &gc) const {
        std::array<int,observation_space_size> ret {};

        int offset = 0;

        ret[offset++] = std::min(gc.curHp, playerHpMax);
        ret[offset++] = std::min(gc.maxHp, playerHpMax);
        ret[offset++] = std::min(gc.gold, playerGoldMax);
        ret[offset++] = gc.floorNum;

        int bossEncodeIdx = offset + bossEncodeMap.at(gc.boss);
        ret[bossEncodeIdx] = 1;
        offset += 10;

        for (auto c : gc.deck.cards) {
            int encodeIdx = offset + getCardIdx(c);
            ret[encodeIdx] = std::min(ret[encodeIdx]+1, cardCountMax);
        }
        offset += 220;

        for (auto r : gc.relics.relics) {
            int encodeIdx = offset + static_cast<int>(r.id);
            ret[encodeIdx] = 1;
        }
        offset += 178;

        return ret;
    }

    std::array<int,NNInterface::observation_space_size> NNInterface::getObservationMaximums() const {
        std::array<int,observation_space_size> ret {};
        int spaceOffset = 0;

        ret[0] = playerHpMax;
        ret[1] = playerHpMax;
        ret[2] = playerGoldMax;
        ret[3] = 60;
        spaceOffset += 3;

        std::fill(ret.begin()+spaceOffset, ret.end(), 1);
        spaceOffset += 10;

        std::fill(ret.begin()+spaceOffset, ret.end(), cardCountMax);
        spaceOffset += 220;

        std::fill(ret.begin()+spaceOffset, ret.end(), 1);
        spaceOffset += 178;

        return ret;
    }

    std::vector<int> NNInterface::createOneHotCardEncodingMap() {
        std::vector<CardId> redCards;
        for (int i = static_cast<int>(CardId::INVALID); i <= static_cast<int>(CardId::ZAP); ++i) {
            auto cid = static_cast<CardId>(i);
            auto color = getCardColor(cid);
            if (color == CardColor::RED) {
                redCards.push_back(cid);
            }
        }

        std::vector<CardId> colorlessCards;
        for (int i = 0; i < srcColorlessCardPoolSize; ++i) {
            colorlessCards.push_back(srcColorlessCardPool[i]);
        }
        std::sort(colorlessCards.begin(), colorlessCards.end(), [](auto a, auto b) {
            return std::string(getCardEnumName(a)) < std::string(getCardEnumName(b));
        });

        std::vector<int> encodingMap(372);
        std::fill(encodingMap.begin(), encodingMap.end(), 0);

        int hotEncodingIdx = 0;
        for (auto x : redCards) {
            encodingMap[static_cast<int>(x)] = hotEncodingIdx++;
        }
        for (auto x : colorlessCards) {
            encodingMap[static_cast<int>(x)] = hotEncodingIdx++;
        }

        return encodingMap;
    }

    std::unordered_map<MonsterEncounter, int> NNInterface::createBossEncodingMap() {
        std::unordered_map<MonsterEncounter, int> bossMap;
        bossMap[ME::SLIME_BOSS] = 0;
        bossMap[ME::HEXAGHOST] = 1;
        bossMap[ME::THE_GUARDIAN] = 2;
        bossMap[ME::CHAMP] = 3;
        bossMap[ME::AUTOMATON] = 4;
        bossMap[ME::COLLECTOR] = 5;
        bossMap[ME::TIME_EATER] = 6;
        bossMap[ME::DONU_AND_DECA] = 7;
        bossMap[ME::AWAKENED_ONE] = 8;
        bossMap[ME::THE_HEART] = 9;
        return bossMap;
    }

    NNInterface* NNInterface::getInstance() {
        if (theInstance == nullptr) {
            theInstance = new NNInterface;
        }
        return theInstance;
    }

}

namespace sts::py {

    void play() {
        sts::SimulatorContext ctx;
        sts::ConsoleSimulator sim;
        sim.play(std::cin, std::cout, ctx);
    }

    search::ScumSearchAgent2* getAgent() {
        static search::ScumSearchAgent2 *agent = nullptr;
        if (agent == nullptr) {
            agent = new search::ScumSearchAgent2();
            agent->pauseOnCardReward = true;
        }
        return agent;
    }

    void playout(GameContext &gc) {
        auto agent = getAgent();
        agent->playout(gc);
    }

    std::vector<Card> getCardReward(GameContext &gc) {
        const bool inValidState = gc.outcome == GameOutcome::UNDECIDED &&
                                  gc.screenState == ScreenState::REWARDS &&
                                  gc.info.rewardsContainer.cardRewardCount > 0;

        if (!inValidState) {
            std::cerr << "GameContext was not in a state with card rewards, check that the game has not completed first." << std::endl;
            return {};
        }

        const auto &r = gc.info.rewardsContainer;
        const auto &cardList = r.cardRewards[r.cardRewardCount-1];
        return std::vector<Card>(cardList.begin(), cardList.end());
    }

    void pickRewardCard(GameContext &gc, Card card) {
        const bool inValidState = gc.outcome == GameOutcome::UNDECIDED &&
                                  gc.screenState == ScreenState::REWARDS &&
                                  gc.info.rewardsContainer.cardRewardCount > 0;
        if (!inValidState) {
            std::cerr << "GameContext was not in a state with card rewards, check that the game has not completed first." << std::endl;
            return;
        }
        auto &r = gc.info.rewardsContainer;
        gc.deck.obtain(gc, card);
        r.removeCardReward(r.cardRewardCount-1);
    }

    void skipRewardCards(GameContext &gc) {
        const bool inValidState = gc.outcome == GameOutcome::UNDECIDED &&
                                  gc.screenState == ScreenState::REWARDS &&
                                  gc.info.rewardsContainer.cardRewardCount > 0;
        if (!inValidState) {
            std::cerr << "GameContext was not in a state with card rewards, check that the game has not completed first." << std::endl;
            return;
        }

        if (gc.hasRelic(RelicId::SINGING_BOWL)) {
            gc.playerIncreaseMaxHp(2);
        }

        auto &r = gc.info.rewardsContainer;
        r.removeCardReward(r.cardRewardCount-1);
    }



    // BEGIN MAP THINGS ****************************

    std::vector<int> getNNMapRepresentation(const Map &map) {
        std::vector<int> ret;

        // 7 bits
        // push edges to first row
        for (int x = 0; x < 7; ++x) {
            if (map.getNode(x,0).edgeCount > 0) {
                ret.push_back(true);
            } else {
                ret.push_back(false);
            }
        }

        // for each node in a row, push valid edges to next row, 3 bits per node, 21 bits per row
        // skip 14th row because it is invariant
        // 21 * 13 == 273 bits
        for (int y = 0; y < 14; ++y) {
            for (int x = 0; x < 7; ++x) {

                bool localEdgeValues[3] {false, false, false};
                auto node = map.getNode(x,y);
                for (int i = 0; i < node.edgeCount; ++i) {
                    auto edge = node.edges[i];
                    if (edge < x) {
                        localEdgeValues[0] = true;
                    } else if (edge == x) {
                        localEdgeValues[1] = true;
                    } else {
                        localEdgeValues[2] = true;
                    }
                }
                ret.insert(ret.end(), localEdgeValues, localEdgeValues+3);
            }
        }

        // room types - for each node there are 6 possible rooms,
        // the first row is always monster, the 8th row is always treasure, 14th is always rest
        // this gives 14-3 valid rows == 11
        // 11 * 6 * 7 = 462 bits
        for (int y = 1; y < 14; ++y) {
            if (y == 8) {
                continue;
            }
            for (int x = 0; x < 7; ++x) {
                auto roomType = map.getNode(x,y).room;
                for (int i = 0; i < 6; ++i) {
                    ret.push_back(static_cast<int>(roomType) == i);
                }
            }
        }

        return ret;
    };

    Room getRoomType(const Map &map, int x, int y) {
        if (x < 0 || x > 6 || y < 0 || y > 14) {
            return Room::INVALID;
        }

        return map.getNode(x,y).room;
    }

    bool hasEdge(const Map &map, int x, int y, int x2) {
        if (x == -1) {
            return map.getNode(x2,0).edgeCount > 0;
        }

        if (x < 0 || x > 6 || y < 0 || y > 14) {
            return false;
        }


        auto node = map.getNode(x,y);
        for (int i = 0; i < node.edgeCount; ++i) {
            if (node.edges[i] == x2) {
                return true;
            }
        }
        return false;
    }

}

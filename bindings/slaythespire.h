//
// Created by keega on 9/24/2021.
//

#ifndef STS_LIGHTSPEED_SLAYTHESPIRE_H
#define STS_LIGHTSPEED_SLAYTHESPIRE_H

#include <vector>
#include <unordered_map>
#include <array>

#include "combat/MonsterGroup.h"
#include "combat/Player.h"
#include "constants/MonsterEncounters.h"
#include "constants/Rooms.h"

namespace sts {

    struct NNInterface {
        static constexpr int observation_space_size = 412;
        static constexpr int playerHpMax = 200;
        static constexpr int playerGoldMax = 1800;
        static constexpr int cardCountMax = 7;

        const std::vector<int> cardEncodeMap;
        const std::unordered_map<MonsterEncounter, int> bossEncodeMap;

        static inline NNInterface *theInstance = nullptr;

        NNInterface();

        int getCardIdx(Card c) const;
        std::array<int,observation_space_size> getObservationMaximums() const;
        std::array<int,observation_space_size> getObservation(const GameContext &gc) const;


        static std::vector<int> createOneHotCardEncodingMap();
        static std::unordered_map<MonsterEncounter, int> createBossEncodingMap();
        static NNInterface* getInstance();

    };

    struct RLInterface{
        static std::array<float, 214> getStateEmbedding(GameContext &gc, BattleContext &bc);
        static std::array<float, 22> getMonsterEmbedding(Monster &monster);
        static std::array<float, 100> getPlayerEmbedding(Player &player);
        static void prettyPrintStateEmbedding(GameContext &gc, BattleContext &bc);
        static std::vector<MonsterEncounter> getImplementedMonsterEncounters();
    };

    namespace search {
        class ScumSearchAgent2;
    }


    class GameContext;
    class Map;

    namespace py {

        void play();

        search::ScumSearchAgent2* getAgent();
        void setGc(const GameContext &gc);
        GameContext* getGc();

        void playout();
        std::vector<Card> getCardReward(GameContext &gc);
        void pickRewardCard(GameContext &gc, Card card);
        void skipRewardCards(GameContext &gc);

        std::vector<int> getNNMapRepresentation(const Map &map);
        Room getRoomType(const Map &map, int x, int y);
        bool hasEdge(const Map &map, int x, int y, int x2);
    }


}


#endif //STS_LIGHTSPEED_SLAYTHESPIRE_H

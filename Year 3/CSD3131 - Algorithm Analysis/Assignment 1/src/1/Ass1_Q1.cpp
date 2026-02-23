
#include <vector>
#include <string>
#include <iostream>
#include <deque>
#include <algorithm>
#include <random>
#include <numeric>

struct Instance {
    std::vector<std::string> servicemenNames;           // e.g., S1,S2...
    std::vector<std::string> vocationNames;             // e.g., Infantry, Armor, Cyber, Artillery, Technician
    std::vector<std::vector<int>> servicemanPrefs;      // servicemanPrefs[s] = ordered list of vocation by serviceman
    std::vector<std::vector<int>> vocationPrefs;        // vocationPrefs[v]   = ordered list of serviceman by SAF
    std::vector<int> capacity;                          // total intake
    std::string label;                                  // scenario name
};

static std::vector<std::vector<int>> buildVocationRank(const std::vector<std::vector<int>>& vocationPrefs, int S, int V) {
    std::vector<std::vector<int>> rank(V, std::vector<int>(S, INT_MAX));
    for (int v = 0; v < V; ++v) {
        for (int pos = 0; pos < (int)vocationPrefs[v].size(); ++pos) {

            // rank[v][s] = position of serviceman s in vocation v's preference list (smaller is better)
            rank[v][ vocationPrefs[v][pos] ] = pos;
        }
    }
    return rank;
}

/**
 * @brief Many-to-one Gale–Shapley (serviceman-proposing).
 * @param S num of servicemen
 * @param V num of vocations
 * @param servicemanPrefs pref lists of servicemen
 * @param vocationPrefs pref lists of vocations
 * @param capacity capacity per vocation
 * @return assignment = list of servicemen assigned to vocation v
 */
 std::vector< std::vector<int>> galeShapleyCollegeAdmissions(
    int S, int V,
    const  std::vector< std::vector<int>>& servicemanPrefs,
    const  std::vector< std::vector<int>>& vocationPrefs,
    const  std::vector<int>& capacity
) {
     std::vector< std::vector<int>> assigned(V);                      // current admits for each vocation
     std::vector< std::vector<int>> rank = buildVocationRank(vocationPrefs, S, V);

    // Next index to propose for each serviceman
     std::vector<int> nextIdx(S, 0);

    // All servicemen start free
    std::deque<int> freeQueue;
    for (int s = 0; s < S; ++s) freeQueue.push_back(s);

    while (!freeQueue.empty()) {
        int s = freeQueue.front();
        freeQueue.pop_front();

        // If s has no more vocations to propose to, they remain unmatched
        if (nextIdx[s] >= (int)servicemanPrefs[s].size()) {
            continue;
        }

        int v = servicemanPrefs[s][ nextIdx[s]++ ];  // propose to next vocation

        // If vocation has zero capacity, immediate rejection
        if (capacity[v] == 0) {
            freeQueue.push_back(s);
            continue;
        }

        if ((int)assigned[v].size() < capacity[v]) {
            // Space available: tentatively accept
            assigned[v].push_back(s);
        } else {
            // Full: find worst current admit by vocation's ranking
            int worst = assigned[v][0];
            for (int x : assigned[v]) {
                if (rank[v][x] > rank[v][worst]) worst = x;
            }
            // If v prefers s over worst, swap
            if (rank[v][s] < rank[v][worst]) {
                // remove worst
                auto it = find(assigned[v].begin(), assigned[v].end(), worst);
                if (it != assigned[v].end()) assigned[v].erase(it);
                assigned[v].push_back(s);
                // worst becomes free
                freeQueue.push_back(worst);
            } else {
                // Rejected; s remains free to try next choice
                freeQueue.push_back(s);
            }
        }
    }

    return assigned;
}

/**
 * @brief Check stability: no blocking pair (s, v).
 * A pair (s, v) blocks if, s prefer v over curr vocation AND v got spare or prefer S over its current admitted S
 */
std::pair<bool, std::vector<std::pair<int,int>>> isStable(
    int S, int V,
    const std::vector<std::vector<int>>& servicemanPrefs,
    const std::vector<std::vector<int>>& vocationPrefs,
    const std::vector<int>& capacity,
    const std::vector<std::vector<int>>& assigned
) {
    // Build inverse map: s -> current vocation (or -1)
    std::vector<int> sToV(S, -1);
    for (int v = 0; v < V; ++v) {
        for (int s : assigned[v]) sToV[s] = v;
    }

    std::vector<std::vector<int>> vRank = buildVocationRank(vocationPrefs, S, V);

    // Build quick rank map for servicemen over vocations
    std::vector<std::vector<int>> sRank(S);
    for (int s = 0; s < S; ++s) {
        sRank[s] = std::vector<int>(V, INT_MAX);
        for (int pos = 0; pos < (int)servicemanPrefs[s].size(); ++pos) {
            sRank[s][ servicemanPrefs[s][pos] ] = pos;
        }
    }

    std::vector<std::pair<int,int>> blocking;

    for (int s = 0; s < S; ++s) {
        int cur = sToV[s]; // -1 if no match
        // Compute set of vocations s prefers over current assignment
        std::vector<int> better;
        if (cur == -1) {
            // s prefers any listed vocation (over being no match)
            better = servicemanPrefs[s];
        } else {
            for (int v : servicemanPrefs[s]) {
                if (sRank[s][v] < sRank[s][cur]) better.push_back(v);
            }
        }

        for (int v : better) {
            const std::vector<int>& list = assigned[v];
            int cap = capacity[v];

            // If v has spare positive capacity, it can accept s now -> blocking
            if (cap > 0 && (int)list.size() < cap) {
                blocking.emplace_back(s, v);
                continue;
            }
            // If cap == 0 or empty with cap==0, cannot accept anyone -> no block via swap
            if (cap == 0 || list.empty()) {
                continue;
            }

            // Otherwise compare with v's worst current admit
            int worst = list[0];
            for (int x : list) if (vRank[v][x] > vRank[v][worst]) worst = x;

            if (vRank[v][s] < vRank[v][worst]) {
                blocking.emplace_back(s, v);
            }
        }
    }

    return {blocking.empty(), blocking};
}

static void printMatching(const Instance& I, const std::vector<std::vector<int>>& assigned) {
    std::cout << "Assignment (Vocation -> Servicemen):\n";
    for (int v = 0; v < (int)I.vocationNames.size(); ++v) {
        std::cout << "  " << I.vocationNames[v] << " [cap=" << I.capacity[v] << "]: ";
        bool first = true;
        for (int s : assigned[v]) {
            if (!first){
            std::cout << ", ";
            }
            std::cout << I.servicemenNames[s];
            first = false;
        }
        std::cout << "\n";
    }
}

// create identity names S1..Sn, V1..Vm when needed 
static std::vector<std::string> makeNames(const std::string& prefix, int n) {
    std::vector<std::string> r; r.reserve(n);
    for (int i = 1; i <= n; ++i) r.push_back(prefix + std::to_string(i));
    return r;
}

// Deterministic shuffle 
static std::vector<int> perm(int n, uint32_t seed) {
    std::vector<int> a(n); std::iota(a.begin(), a.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(a.begin(), a.end(), rng);
    return a;
}

// Build a “criteria-driven” preference order for servicemen over vocations (score > order)
static std::vector<std::vector<int>> servicemanPrefsFromCriteria(
    int S, int V,
    const std::vector<std::vector<int>>& interestWeight,     // interestWeight[s][v]: higher = better
    const std::vector<std::vector<int>>& distancePenalty,    // distancePenalty[s][v]: higher = worse
    const std::vector<int>& overseasBonus               // overseasBonus[v]: bonus if s wants overseas
) {
    std::vector<std::vector<int>> prefs(S, std::vector<int>());
    for (int s = 0; s < S; ++s) {
        std::vector<std::pair<int,int>> scored; // (-score, v) so we can sort ascending
        for (int v = 0; v < V; ++v) {
            int score = 5 * interestWeight[s][v] - 1 * distancePenalty[s][v] + overseasBonus[v];
            scored.push_back({-score, v});
        }
        std::sort(scored.begin(), scored.end());
        for (auto &p : scored) prefs[s].push_back(p.second);
    }
    return prefs;
}

// Build a “criteria-driven” preference order for vocations over servicemen (SAF POV) (score > order). */
static std::vector<std::vector<int>> vocationPrefsFromCriteria(
    int S, int V,
    const std::vector<int>& pesScore,                 // larger = fitter
    const std::vector<int>& hasRelevantDegreeScore,   // 1 or 0 indicating relevance for this test
    const std::vector<std::vector<int>>& degreeRelevanceByV // degreeRelevanceByV[v][s] (0..3)
) {
    std::vector<std::vector<int>> prefs(V, std::vector<int>());
    for (int v = 0; v < V; ++v) {
        std::vector<std::pair<int,int>> scored; // (-score, s)
        for (int s = 0; s < S; ++s) {
            int score = 4 * pesScore[s] + 3 * degreeRelevanceByV[v][s] + 2 * hasRelevantDegreeScore[s];
            scored.push_back({-score, s});
        }
        sort(scored.begin(), scored.end());
        for (auto &p : scored) prefs[v].push_back(p.second);
    }
    return prefs;
}

// Helper to build a generic random-like instance
static Instance makeRandomInstance(int S, int V, uint32_t seed, const std::string& label, std::vector<int> caps) {
    Instance I;
    I.label = label;
    I.servicemenNames = makeNames("S", S);
    I.vocationNames   = makeNames("V", V);
    I.capacity        = caps;

    // serviceman prefs: random permutations (deterministic)
    for (int s = 0; s < S; ++s) {
        std::vector<int> p = perm(V, seed + 101 * (s+1));
        I.servicemanPrefs.push_back(p);
    }
    // vocation prefs: random permutations
    for (int v = 0; v < V; ++v) {
        std::vector<int> p = perm(S, seed + 777 * (v+1));
        I.vocationPrefs.push_back(p);
    }
    return I;
}

// A criteria-driven instance (reflecting PES/degree/distance/overseas) 
static Instance makeCriteriaInstance_1() {
    // 8 servicemen, 5 vocations
    const int S = 8, V = 5;
    Instance I;
    I.label = "Criteria-driven #1 (PES/Degree/Distance/Overseas)";
    I.servicemenNames = {"Ali","Ben","Chen","Deepak","Ethan","Farid","Xiao Ming","Haziq"};
    I.vocationNames = {"Infantry","Armour","Artillery","CyberDefence","Technician"};
    I.capacity = {2,1,2,2,1}; // total 8

    // Servicemen criteria (toy data)
    // interestWeight[s][v] in {0..3}, distancePenalty[s][v] in {0..5}, overseasBonus[v] in {0,2}
    std::vector<std::vector<int>> interest = {
        /*Ali*/    {3,2,1,2,1},
        /*Ben*/    {2,2,1,3,1},
        /*Chen*/   {1,1,3,2,1},
        /*Deepak*/ {2,3,1,2,1},
        /*Ethan*/  {3,1,2,1,1},
        /*Farid*/  {1,2,3,1,1},
        /*Goh*/    {2,1,2,3,1},
        /*Haziq*/  {1,1,2,3,2}
    };
    std::vector<std::vector<int>> dist = {
        /*Ali*/    {1,2,4,3,2},
        /*Ben*/    {1,3,2,3,3},
        /*Chen*/   {3,4,1,2,3},
        /*Deepak*/ {2,1,3,2,4},
        /*Ethan*/  {3,2,3,2,1},
        /*Farid*/  {2,3,2,3,2},
        /*Goh*/    {2,4,1,1,3},
        /*Haziq*/  {3,2,2,1,2}
    };
    std::vector<int> overseasBonus = {
        /*Infantry*/2, /*Armour*/1, /*Artillery*/3, /*CyberDefence*/0, /*Technician*/0
    };
    I.servicemanPrefs = servicemanPrefsFromCriteria(S, V, interest, dist, overseasBonus);

    // Vocation criteria: PES score (A=4, B1=3, B2=3, B3=2, B4=2, C=1 etc.), degree relevance by unit
    std::vector<int> pesScore        = {4,3,2,3,4,2,3,2};
    std::vector<int> hasRelDegree    = {0,1,1,1,0,0,1,0}; // e.g., IT/Eng = 1 for demo
    // degree relevance matrix per vocation (0..3)
    std::vector<std::vector<int>> degreeRel(V, std::vector<int>(S, 0));
    // Infantry likes high PES; degree relevance small
    for (int s=0;s<S;++s) degreeRel[0][s] = (pesScore[s]>=3?1:0);
    // Armour likes PES + Eng
    for (int s=0;s<S;++s) degreeRel[1][s] = hasRelDegree[s] ? 2 : 0;
    // Artillery overseas vibe,approximate with pes>=3 and some degree
    for (int s=0;s<S;++s) degreeRel[2][s] = (pesScore[s]>=3?1:0) + (hasRelDegree[s]?1:0);
    // CyberDefence likes IT/CS
    for (int s=0;s<S;++s) degreeRel[3][s] = hasRelDegree[s] ? 3 : 0;
    // Technician like Engineering deg
    for (int s=0;s<S;++s) degreeRel[4][s] = hasRelDegree[s] ? 2 : 1;

    I.vocationPrefs = vocationPrefsFromCriteria(S, V, pesScore, hasRelDegree, degreeRel);
    return I;
}

/** Another criteria-driven instance with tighter capacity */
static Instance makeCriteriaInstance_2() {
    // 10 servicemen, 6 vocations, total capacity 8 (some unmatched)
    const int S = 10, V = 6;
    Instance I;
    I.label = "Criteria-driven #2 (Tight capacity)";
    I.servicemenNames = {"A","B","C","D","E","F","G","H","I","J"};
    I.vocationNames   = {"Infantry","Armour","Navy","AirForce","Cyber","Logistics"};
    I.capacity        = {2,1,1,1,2,1}; // total 8

    // For brevity, craft preferences by simple patterned scores
    std::vector<std::vector<int>> interest(S, std::vector<int>(V, 1));
    for (int s=0;s<S;++s){
        // Bias: even indices like Cyber/AirForce; odd like Infantry/Navy
        interest[s][ (s%2==0)?4:0 ] = 3;
        interest[s][ (s%3==0)?3:2 ] = 2;
    }
    std::vector<std::vector<int>> dist(S, std::vector<int>(V, 2));
    for (int s=0;s<S;++s) dist[s][ s%V ] = 0; // “closest” one
    std::vector<int> overseasBonus = {0,0,2,2,0,0};

    I.servicemanPrefs = servicemanPrefsFromCriteria(S,V,interest,dist,overseasBonus);

    std::vector<int> pes(S,3);
    for (int s=0;s<S;++s) if (s%4==0) pes[s]=4; else if (s%5==0) pes[s]=2;
    std::vector<int> hasRel(S,0);
    for (int s=0;s<S;++s) if (s%2==0) hasRel[s]=1; // half have relevant degrees

    std::vector<std::vector<int>> degRel(V, std::vector<int>(S,0));
    // Infantry prefers PES; Cyber prefers degree; AirForce prefers high PES + some degree
    for (int s=0;s<S;++s) {
        degRel[0][s] = (pes[s]>=3?2:0);
        degRel[1][s] = hasRel[s]?1:0;
        degRel[2][s] = (pes[s]>=3?1:0);
        degRel[3][s] = (pes[s]==4?2:0) + (hasRel[s]?1:0);
        degRel[4][s] = hasRel[s]?3:0;
        degRel[5][s] = hasRel[s]?2:1;
    }

    I.vocationPrefs = vocationPrefsFromCriteria(S,V,pes,hasRel,degRel);
    return I;
}

/** Build 10 scenarios (some random/structured), all deterministic */
static std::vector<Instance> buildAllScenarios() {
    std::vector<Instance> L;

    // 1) Criteria-driven baseline with full capacity
    L.push_back(makeCriteriaInstance_1());

    // 2) Criteria-driven tight capacity (some unmatched)
    L.push_back(makeCriteriaInstance_2());

    // 3) Many servicemen want the same top vocation
    {
        Instance I = makeRandomInstance(8, 5, /*seed*/42, "Many want same top vocation", {2,1,2,2,1});
        for (int s=0;s<8;++s) {
            // Move V0 to top for everyone
            std::vector<int>& pref = I.servicemanPrefs[s];
            auto it = find(pref.begin(), pref.end(), 0);
            if (it != pref.end()) {
                pref.erase(it);
                pref.insert(pref.begin(), 0);
            }
        }
        L.push_back(I);
    }

    // 4) A vocation is very picky about a few servicemen
    {
        Instance I = makeRandomInstance(8,5, 77, "One vocation very picky", {2,1,2,1,1});
        // Make V2 prefer S0,S1 strongly
        I.vocationPrefs[2].erase(remove(I.vocationPrefs[2].begin(), I.vocationPrefs[2].end(), 0), I.vocationPrefs[2].end());
        I.vocationPrefs[2].erase(remove(I.vocationPrefs[2].begin(), I.vocationPrefs[2].end(), 1), I.vocationPrefs[2].end());
        I.vocationPrefs[2].insert(I.vocationPrefs[2].begin(), {0,1});
        L.push_back(I);
    }

    // 5) Sparse serviceman preference lists (top-3 only)
    {
        Instance I = makeRandomInstance(8,5, 99, "Sparse serviceman prefs", {2,1,2,1,1});
        for (int s=0;s<8;++s) {
            if ((s%2)==0 && (int)I.servicemanPrefs[s].size()>3)
                I.servicemanPrefs[s].resize(3);
        }
        L.push_back(I);
    }

    // 6) Tight capacity (total seats < servicemen)
    L.push_back(makeRandomInstance(8,5, 123, "Tight capacity (6/8)", {1,1,2,1,1}));

    // 7) Generous capacity (everyone can be placed if they listed enough)
    L.push_back(makeRandomInstance(8,5, 321, "Generous capacity (8/8)", {2,2,2,1,1}));

    // 8) All servicemen rank vocations in the same order (worst-case proposals)
    {
        Instance I = makeRandomInstance(8,5, 555, "All servicemen same order", {2,1,2,1,1});
        for (int s=0;s<8;++s) {
            I.servicemanPrefs[s].clear();
            for (int v=0; v<5; ++v) I.servicemanPrefs[s].push_back(v);
        }
        L.push_back(I);
    }

    // 9) One big vocation (acts like a magnet), one closed (cap 0)
    {
        Instance I = makeRandomInstance(8,5, 2024, "One big vocation; one closed", {4,1,2,1,0});
        L.push_back(I);
    }

    // 10) Some servicemen omit a popular vocation entirely
    {
        Instance I = makeRandomInstance(8,5, 909, "Some omit V2 entirely", {2,1,2,1,1});
        for (int s=0;s<3;++s) {
            std::vector<int>& p = I.servicemanPrefs[s];
            p.erase(remove(p.begin(), p.end(), 2), p.end()); // remove V2
            if (p.empty()) { p = {0,1}; } // ensure non-empty for demo
        }
        L.push_back(I);
    }
    return L;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::vector<Instance> scenarios = buildAllScenarios();

    for (size_t i = 0; i < scenarios.size(); ++i) {
        const Instance& I = scenarios[i];

        int S = (int)I.servicemenNames.size();
        int V = (int)I.vocationNames.size();

        std::cout << "============================\n";
        std::cout << "Case " << (i+1) << ": " << I.label << "\n";
        std::cout << "Servicemen: " << S << ", Vocations: " << V << "\n";
        int totalCap = accumulate(I.capacity.begin(), I.capacity.end(), 0);
        std::cout << "Total capacity: " << totalCap << "\n";

        auto assigned = galeShapleyCollegeAdmissions(S, V, I.servicemanPrefs, I.vocationPrefs, I.capacity);
        printMatching(I, assigned);

        auto [stable, blocking] = isStable(S, V, I.servicemanPrefs, I.vocationPrefs, I.capacity, assigned);
        std::cout << "Stable? " << (stable ? "YES" : "NO") << "\n";
        if (!stable) {
            std::cout << "Blocking pairs (up to 10 shown):\n";
            for (size_t k = 0; k < blocking.size() && k < 10; ++k) {
                int s = blocking[k].first, v = blocking[k].second;
                std::cout << "  (" << I.servicemenNames[s] << ", " << I.vocationNames[v] << ")\n";
            }
        }
        std::cout << "\n";
    }

    return 0;
}
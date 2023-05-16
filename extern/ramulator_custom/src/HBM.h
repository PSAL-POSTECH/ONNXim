#ifndef __HBM_H
#define __HBM_H

#include <map>
#include <vector>
#include <string>
#include <functional>

#include "Request.h"

using namespace std;

namespace ram
{
template <typename T>
class DRAM;

class HBM
{
public:
    static string standard_name;
    enum class Org;
    enum class Speed;
    HBM(Org org, Speed speed);
    HBM(const string& org_str, const string& speed_str);

    static map<string, enum Org> org_map;
    static map<string, enum Speed> speed_map;

    /* Level */
    enum class Level : int
    {
        Channel, Rank, BankGroup, Bank, Row, Column, MAX
    };
    
    static std::string level_str [int(Level::MAX)];

    /* Command */
    enum class Command : int
    {
        ACT, PRE,   PREA,
        RD,  WR,    PIM_WR, RDA, WRA,
        REF, REFSB, PDE, PDX,  SRE, SRX,
        MAX
    };

    // REFSB and REF is not compatible, choose one or the other.
    // REFSB can be issued to banks in any order, as long as REFI1B
    // is satisfied for all banks

    string command_name[int(Command::MAX)] = {
        "ACT", "PRE",   "PREA",
        "RD",  "WR",    "PIM_WR", "RDA",  "WRA",
        "REF", "REFSB", "PDE",  "PDX",  "SRE", "SRX"
    };

    Level scope[int(Command::MAX)] = {
        Level::Row,    Level::Bank,   Level::Rank,
        Level::Column, Level::Column, Level::Column, Level::Column, Level::Column,
        Level::Rank,   Level::Bank,   Level::Rank,   Level::Rank,   Level::Rank,   Level::Rank
    };

    bool is_opening(Command cmd)
    {
        switch(int(cmd)) {
            case int(Command::ACT):
                return true;
            default:
                return false;
        }
    }

    bool is_accessing(Command cmd)
    {
        switch(int(cmd)) {
            case int(Command::RD):
            case int(Command::WR):
            case int(Command::RDA):
            case int(Command::WRA):
                return true;
            default:
                return false;
        }
    }

    bool is_closing(Command cmd)
    {
        switch(int(cmd)) {
            case int(Command::RDA):
            case int(Command::WRA):
            case int(Command::PRE):
            case int(Command::PREA):
                return true;
            default:
                return false;
        }
    }

    bool is_refreshing(Command cmd)
    {
        switch(int(cmd)) {
            case int(Command::REF):
            case int(Command::REFSB):
                return true;
            default:
                return false;
        }
    }

    /* State */
    enum class State : int
    {
        Opened, Closed, PowerUp, ActPowerDown, PrePowerDown, SelfRefresh, MAX
    } start[int(Level::MAX)] = {
        State::MAX, State::PowerUp, State::MAX, State::Closed, State::Closed, State::MAX
    };

    /* Translate */
    Command translate[int(Request::Type::MAX)] = {
        Command::RD,  Command::WR, Command::PIM_WR,
        Command::REF, Command::PDE, Command::SRE
    };

    /* Prereq */
    function<Command(DRAM<HBM>*, Command cmd, int)> prereq[int(Level::MAX)][int(Command::MAX)];

    // SAUGATA: added function object container for row hit status
    /* Row hit */
    function<bool(DRAM<HBM>*, Command cmd, int)> rowhit[int(Level::MAX)][int(Command::MAX)];
    function<bool(DRAM<HBM>*, Command cmd, int)> rowopen[int(Level::MAX)][int(Command::MAX)];

    /* Timing */
    struct TimingEntry
    {
        Command cmd;
        int dist;
        int val;
        bool sibling;
    };
    vector<TimingEntry> timing[int(Level::MAX)][int(Command::MAX)];

    /* Lambda */
    function<void(DRAM<HBM>*, int)> lambda[int(Level::MAX)][int(Command::MAX)];

    /* Organization */
    enum class Org : int
    { // per channel density here. Each stack comes with 8 channels
        HBM_1Gb,
        HBM_2Gb,
        HBM_4Gb,
        MAX
    };

    struct OrgEntry {
        int size;
        int dq;
        int count[int(Level::MAX)];
    } org_table[int(Org::MAX)] = {
        {1<<10, 128, {0, 0, 4, 2, 1<<20, 1<<(6+1)}},
        {2<<10, 128, {0, 0, 4, 2, 1<<20, 1<<(6+1)}},
        {4<<10, 128, {0, 0, 4, 4, 1<<20, 1<<(6+1)}},
    }, org_entry;

    void set_channel_number(int channel);
    void set_rank_number(int rank);

    /* Speed */
    enum class Speed : int
    {
        HBM_1Gbps,
        HBM_2Gbps,
        MAX
    };

    int prefetch_size = 2; // burst length could be 2 and 4 (choose 4 here), 2n prefetch
    int channel_width = 128;

    struct SpeedEntry {
        int rate;
        double freq, tCK;
        int nBL, nCCDS, nCCDL;
        int nCL, nRCDR, nRCDW, nRP, nCWL;
        int nRAS, nRC;
        int nRTP, nWTRS, nWTRL, nWR;
        int nRRDS, nRRDL, nFAW;
        int nRFC, nREFI, nREFI1B;
        int nPD, nXP;
        int nCKESR, nXS;
    } speed_table[int(Speed::MAX)] = {
        {1000, // rate
         500, 2.0, // freq, tCK
         // FIX: Why is nBL set to 2 instead of 1?
         // FIX: It seems that this is because a single request corresponds to 64B,
         // ,which means that `prefetch_size = 4`.
         1, 1, 2, // nBL, nCCDS, nCCDL
         7, 7, 6, 7, 4, // nCL, nRCDR, nRCDW, nRP, nCWL
         17, 24, // nRAS, nRC
         7, 2, 4, 8, // nRTP, nWTRS, nWTRL, nWR
         4, 5, 20, // nRRDS, nRRDL, nFAW
         0, 1950, 0, // nRFC, nREFI, nREFI1B
         5, 5,  // nPD, nXP
         5, 0 },  // nCKESR, nXS
        {2000, 
         1000, 1.0, 
         1, 1, 2, 
         14, 14, 12, 14, 8, 
         34, 48, 
         14, 4, 8, 16, 
         8, 10, 40, 
         0, 3900, 0, 
         10, 10, 
         10, 0},
    }, speed_entry;

    int read_latency;

private:
    void init_speed();
    void init_lambda();
    void init_prereq();
    void init_rowhit();  // SAUGATA: added function to check for row hits
    void init_rowopen();
    void init_timing();
};

} /*namespace ram*/

#endif /*__HBM_H*/

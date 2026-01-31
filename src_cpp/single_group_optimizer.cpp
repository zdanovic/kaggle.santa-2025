// Source: Kaggle notebook "a-bit-better-public" by daniil1423
// Adapted for local/Kaggle reproducible use in this repo (unchanged logic).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

constexpr int MAX_N = 200;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;

alignas(64) const long double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
alignas(64) const long double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

struct FastRNG {
    uint64_t s[2];
    FastRNG(uint64_t seed = 42) {
        s[0] = seed ^ 0x853c49e6748fea9bULL;
        s[1] = (seed * 0x9e3779b97f4a7c15ULL) ^ 0xc4ceb9fe1a85ec53ULL;
    }
    inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        uint64_t s0 = s[0], s1 = s[1], r = s0 + s1;
        s1 ^= s0; s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); s[1] = rotl(s1, 37);
        return r;
    }
    inline long double rf() { return (next() >> 11) * 0x1.0p-53L; }
    inline long double rf2() { return rf() * 2.0L - 1.0L; }
    inline int ri(int n) { return next() % n; }
    inline long double gaussian() {
        long double u1 = rf() + 1e-10L, u2 = rf();
        return sqrtl(-2.0L * logl(u1)) * cosl(2.0L * PI * u2);
    }
};

struct Poly {
    long double px[NV], py[NV];
    long double x0, y0, x1, y1;
};

inline void getPoly(long double cx, long double cy, long double deg, Poly& q) {
    long double rad = deg * (PI / 180.0L);
    long double s = sinl(rad), c = cosl(rad);
    long double minx = 1e9L, miny = 1e9L, maxx = -1e9L, maxy = -1e9L;
    for (int i = 0; i < NV; i++) {
        long double x = TX[i] * c - TY[i] * s + cx;
        long double y = TX[i] * s + TY[i] * c + cy;
        q.px[i] = x; q.py[i] = y;
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
    }
    q.x0 = minx; q.y0 = miny; q.x1 = maxx; q.y1 = maxy;
}

inline bool pip(long double px, long double py, const Poly& q) {
    bool in = false;
    int j = NV - 1;
    for (int i = 0; i < NV; i++) {
        if ((q.py[i] > py) != (q.py[j] > py) &&
            px < (q.px[j] - q.px[i]) * (py - q.py[i]) / (q.py[j] - q.py[i]) + q.px[i])
            in = !in;
        j = i;
    }
    return in;
}

inline long double cross(long double ax, long double ay, long double bx, long double by, long double cx, long double cy) {
    return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
}

inline bool segInt(long double ax, long double ay, long double bx, long double by,
                   long double cx, long double cy, long double dx, long double dy) {
    long double d1 = cross(cx, cy, dx, dy, ax, ay);
    long double d2 = cross(cx, cy, dx, dy, bx, by);
    long double d3 = cross(ax, ay, bx, by, cx, cy);
    long double d4 = cross(ax, ay, bx, by, dx, dy);
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
        ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;
    return false;
}

inline bool polyIntersect(const Poly& a, const Poly& b) {
    if (a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0) return false;
    for (int i = 0; i < NV; i++) {
        int ni = (i + 1) % NV;
        for (int j = 0; j < NV; j++) {
            int nj = (j + 1) % NV;
            if (segInt(a.px[i], a.py[i], a.px[ni], a.py[ni], b.px[j], b.py[j], b.px[nj], b.py[nj])) return true;
        }
    }
    if (pip(a.px[0], a.py[0], b) || pip(b.px[0], b.py[0], a)) return true;
    return false;
}

struct Tree {
    long double x, y, deg;
};

struct Cfg {
    int n;
    vector<Tree> t;
    vector<Poly> p;

    Cfg() {}
    Cfg(int n) : n(n) {
        t.resize(n);
        p.resize(n);
    }

    inline void upd(int i) { getPoly(t[i].x, t[i].y, t[i].deg, p[i]); }
    inline void updAll() { for (int i = 0; i < n; i++) upd(i); }

    inline bool anyOvl() const {
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (polyIntersect(p[i], p[j])) return true;
        return false;
    }

    inline long double score() const {
        long double minx = 1e9L, miny = 1e9L, maxx = -1e9L, maxy = -1e9L;
        for (int i = 0; i < n; i++) {
            minx = min(minx, p[i].x0); miny = min(miny, p[i].y0);
            maxx = max(maxx, p[i].x1); maxy = max(maxy, p[i].y1);
        }
        long double s = max(maxx - minx, maxy - miny);
        return (s * s) / (long double)n;
    }
};

inline Cfg copyCfg(const Cfg& c) {
    Cfg o(c.n);
    o.t = c.t;
    o.p = c.p;
    return o;
}

inline long double clamp(long double v, long double lo, long double hi) {
    return max(lo, min(hi, v));
}

inline void perturb(Cfg& c, FastRNG& rng, long double move_scale, long double ang_scale) {
    int i = rng.ri(c.n);
    long double nx = c.t[i].x + rng.rf2() * move_scale;
    long double ny = c.t[i].y + rng.rf2() * move_scale;
    long double nd = fmodl(c.t[i].deg + rng.rf2() * ang_scale + 360.0L, 360.0L);
    c.t[i].x = clamp(nx, -100.0L, 100.0L);
    c.t[i].y = clamp(ny, -100.0L, 100.0L);
    c.t[i].deg = nd;
    c.upd(i);
}

inline Cfg optimizeOne(const Cfg& base, int iters, uint64_t seed) {
    FastRNG rng(seed);
    Cfg cur = copyCfg(base);
    cur.updAll();
    Cfg best = copyCfg(cur);
    long double bestScore = cur.score();

    long double move0 = 0.08L;
    long double ang0 = 20.0L;

    for (int i = 0; i < iters; i++) {
        Cfg cand = copyCfg(cur);
        long double t = (long double)i / (long double)iters;
        long double move = move0 * (1.0L - t * 0.85L);
        long double ang = ang0 * (1.0L - t * 0.85L);
        perturb(cand, rng, move, ang);
        if (cand.anyOvl()) continue;
        long double sc = cand.score();
        if (sc < bestScore) {
            bestScore = sc;
            best = copyCfg(cand);
            cur = cand;
        } else {
            cur = cand;
        }
    }
    return best;
}

inline Cfg optimizeParallel(const Cfg& base, int iters, int restarts) {
    vector<Cfg> bests(restarts);
    vector<long double> scores(restarts, 1e18L);

    #pragma omp parallel for
    for (int r = 0; r < restarts; r++) {
        Cfg o = optimizeOne(base, iters, 1234ULL + (uint64_t)r * 7777ULL);
        bests[r] = o;
        scores[r] = o.score();
    }
    int bestIdx = min_element(scores.begin(), scores.end()) - scores.begin();
    return bests[bestIdx];
}

static inline long double parseVal(const string& s) {
    if (!s.empty() && s[0] == 's') return stold(s.substr(1));
    return stold(s);
}

unordered_map<int, Cfg> loadCSV(const string& path) {
    unordered_map<int, Cfg> res;
    ifstream in(path);
    string line;
    getline(in, line);
    vector<vector<string>> rows;
    while (getline(in, line)) {
        stringstream ss(line);
        vector<string> cols;
        string item;
        while (getline(ss, item, ',')) cols.push_back(item);
        if (cols.size() < 4) continue;
        rows.push_back(cols);
    }
    unordered_map<int, int> cnt;
    for (auto& r : rows) {
        int g = stoi(r[0].substr(0, 3));
        cnt[g]++;
    }
    for (auto& kv : cnt) {
        res[kv.first] = Cfg(kv.second);
    }
    unordered_map<int, int> idx;
    for (auto& r : rows) {
        int g = stoi(r[0].substr(0, 3));
        int i = idx[g]++;
        res[g].t[i].x = parseVal(r[1]);
        res[g].t[i].y = parseVal(r[2]);
        res[g].t[i].deg = parseVal(r[3]);
    }
    for (auto& kv : res) kv.second.updAll();
    return res;
}

void saveCSV(const string& path, unordered_map<int, Cfg>& cfg) {
    ofstream out(path);
    out << "id,x,y,deg\n";
    for (int n = 1; n <= MAX_N; n++) {
        auto it = cfg.find(n);
        if (it == cfg.end()) continue;
        auto& c = it->second;
        for (int i = 0; i < c.n; i++) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%03d_%d", n, i);
            out << buf << ",";
            out << "s" << setprecision(18) << fixed << (double)c.t[i].x << ",";
            out << "s" << setprecision(18) << fixed << (double)c.t[i].y << ",";
            out << "s" << setprecision(18) << fixed << (double)c.t[i].deg << "\n";
        }
    }
}

int main(int argc, char** argv) {
    string in="submission.csv", out="submission_optimized.csv";
    int iters=50000, restarts=64; // Increased defaults for stronger optimization

    // Get group number from environment variable
    const char* groupEnv = getenv("GROUP_NUMBER");
    if (!groupEnv) {
        printf("Error: GROUP_NUMBER environment variable not set\n");
        return 1;
    }
    int targetN = stoi(groupEnv);

    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a=="-i" && i+1<argc) in=argv[++i];
        else if (a=="-o" && i+1<argc) out=argv[++i];
        else if (a=="-n" && i+1<argc) iters=stoi(argv[++i]);
        else if (a=="-r" && i+1<argc) restarts=stoi(argv[++i]);
    }

    auto cfg = loadCSV(in);
    if (cfg.empty() || !cfg.count(targetN)) return 1;

    Cfg c = cfg[targetN];
    long double os = c.score();
    printf("Target N=%d | Initial Score: %.12Lf\n", targetN, os);

    // Dynamic params based on N size
    int it = iters, r = restarts;
    if (targetN <= 10) { it = (int)(iters * 3.0); r = restarts * 3; }
    else if (targetN <= 30) { it = (int)(iters * 2.0); r = (int)(restarts * 2.0); }
    else if (targetN > 100) { it = (int)(iters * 0.8); r = max(8, (int)(restarts * 0.8)); }

    Cfg o = optimizeParallel(c, it, max(8, r));

    long double ns = o.score();
    
    // Save only if strictly better and no overlaps
    if (!o.anyOvl() && ns < os - 1e-12L) {
        printf(">>> IMPROVED N=%d: %.12Lf -> %.12Lf (%.5Lf%%)\n", targetN, os, ns, (os-ns)/os*100.0L);
        cfg[targetN] = o;
        saveCSV(out, cfg); // Save to temp file
    } else {
        printf("No improvement for N=%d\n", targetN);
    }
    return 0;
}

// Resumable exact-EMD center selection and single-copy TQ8 bucket construction.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include "mvsic/algorithms/emdivf/index.h"
#include "mvsic/algorithms/emdivf/emd.h"
#include "mvsic/core/distance_measures/one_to_many.h"
#include "mvsic/core/quantization/turboquant_8bit_mv.h"
#include "parlay/primitives.h"

namespace fs=std::filesystem;
using Set=emdivf::CloudSet;
using Index=emdivf::Index;
using Neighbor=emdivf::Neighbor;
using TQ8=mvsic::turboquant_8bit_mv::Model<false>;
using Clock=std::chrono::steady_clock;
using Args=std::map<std::string,std::string>;
double Seconds(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}
std::string Get(const Args& a,const std::string& key,const std::string& fallback="") {
  auto it=a.find(key); if(it!=a.end()) return it->second;
  if(fallback.empty()) throw std::invalid_argument("missing --"+key); return fallback;
}
template<class T> void Put(std::ostream& f,T x) {
  f.write(reinterpret_cast<const char*>(&x),sizeof(x));
  if(!f) throw std::runtime_error("write failed");
}
template<class T> T Read(std::istream& f) {
  T x{}; f.read(reinterpret_cast<char*>(&x),sizeof(x));
  if(!f) throw std::runtime_error("truncated checkpoint"); return x;
}
bool Better(const Neighbor& a,const Neighbor& b) {
  return a.distance<b.distance || (a.distance==b.distance && a.document<b.document);
}
void Top(std::vector<Neighbor>& a,size_t k) {
  k=std::min(k,a.size());
  if(k<a.size()) std::nth_element(a.begin(),a.begin()+k,a.end(),Better);
  a.resize(k); std::sort(a.begin(),a.end(),Better);
}
std::string Identity(const std::string& path) {
  return fs::canonical(path).string()+"|"+std::to_string(fs::file_size(path))+"|"+
      std::to_string(fs::last_write_time(path).time_since_epoch().count());
}
void Provenance(const std::string& path,const std::string& expected) {
  if(fs::exists(path)) {
    std::ifstream f(path); std::stringstream text; text<<f.rdbuf();
    if(text.str()!=expected) throw std::runtime_error("checkpoint provenance mismatch: "+path);
  } else {
    std::ofstream f(path+".tmp"); f<<expected; f.close();
    if(!f) throw std::runtime_error("provenance write failed");
    fs::rename(path+".tmp",path);
  }
}
void SaveSelection(const Index& x,const std::string& path) {
  std::ofstream f(path+".tmp",std::ios::binary);
  Put<uint64_t>(f,0x32434553444d45ULL); Put(f,x.dimension); Put(f,x.train_size); Put(f,x.seed);
  Put<uint32_t>(f,x.centers.size());
  for(size_t i=0;i<x.centers.size();++i) {
    Put(f,x.centers[i]); Put(f,x.selection_seconds[i]); Put(f,x.cover_radius[i]);
  }
  for(size_t i=0;i<x.train_size;++i) {Put(f,x.train_distance[i]);Put(f,x.train_nearest_center[i]);}
  f.close(); if(!f) throw std::runtime_error("selection checkpoint close failed");
  fs::rename(path+".tmp",path);
}
Index LoadSelection(const std::string& path) {
  std::ifstream f(path,std::ios::binary); Index x;
  if(Read<uint64_t>(f)!=0x32434553444d45ULL) throw std::runtime_error("invalid selection state");
  x.dimension=Read<uint32_t>(f); x.train_size=Read<uint32_t>(f); x.seed=Read<uint64_t>(f);
  const auto n=Read<uint32_t>(f);
  if(!n || n>x.train_size) throw std::runtime_error("invalid center count");
  std::vector<uint8_t> seen(x.train_size);
  for(uint32_t i=0;i<n;++i) {
    auto c=Read<uint32_t>(f);
    if(c>=x.train_size || seen[c]) throw std::runtime_error("invalid center ID");
    seen[c]=1; x.centers.push_back(c);
    x.selection_seconds.push_back(Read<double>(f));x.cover_radius.push_back(Read<double>(f));
  }
  for(uint32_t i=0;i<x.train_size;++i) {
    x.train_distance.push_back(Read<double>(f));x.train_nearest_center.push_back(Read<uint32_t>(f));
    if(!std::isfinite(x.train_distance.back()) || x.train_distance.back()<0 || x.train_nearest_center.back()>=n)
      throw std::runtime_error("invalid nearest-center state");
  }
  return x;
}
void Select(const Args& a) {
  const auto source=Get(a,"train"), path=Get(a,"state");
  if(!fs::path(path).parent_path().empty()) fs::create_directories(fs::path(path).parent_path());
  const uint32_t m=std::stoul(Get(a,"m")); const uint64_t seed=std::stoull(Get(a,"seed","42"));
  const bool bounded=Get(a,"bounded","1")=="1";
  Provenance(path+".source",Identity(source)+"\nseed="+std::to_string(seed)+"\n");
  Set train(source.c_str(),true); Index x;
  if(!m || m>train.size()) throw std::invalid_argument("M exceeds training pool");
  if(fs::exists(path)) x=LoadSelection(path);
  else {
    x.dimension=train.get_dims();x.train_size=train.size();x.seed=seed;
    x.train_distance.assign(train.size(),std::numeric_limits<double>::infinity());
    x.train_nearest_center.assign(train.size(),0);
  }
  if(x.dimension!=train.get_dims() || x.train_size!=train.size() || x.seed!=seed)
    throw std::runtime_error("selection source mismatch");
  std::vector<uint8_t> chosen(train.size()); for(auto c:x.centers) chosen[c]=1;
  std::mt19937_64 rng(seed);
  const uint32_t first=std::uniform_int_distribution<uint32_t>(0,train.size()-1)(rng);
  const double elapsed=x.selection_seconds.empty()?0:x.selection_seconds.back();
  const auto start=Clock::now();
  for(uint32_t c=x.centers.size();c<m;++c) {
    uint32_t next=first;
    if(c) {double farthest=-1;for(uint32_t q=0;q<train.size();++q)
      if(!chosen[q] && x.train_distance[q]>farthest) {farthest=x.train_distance[q];next=q;}}
    const auto center=train[next]; chosen[next]=1;x.centers.push_back(next);
    parlay::parallel_for(0,train.size(),[&](size_t q) {
      if(chosen[q]) return;
      const auto item=train[q];
      const double d=bounded ? emdivf::EmdCapped(item.data(),item.size(),center.data(),center.size(),
          x.dimension,x.train_distance[q]) : emdivf::Emd(item.data(),item.size(),center.data(),center.size(),x.dimension);
      if(d<x.train_distance[q]) {x.train_distance[q]=d;x.train_nearest_center[q]=c;}
    });
    x.train_distance[next]=0;x.train_nearest_center[next]=c;
    x.cover_radius.push_back(*std::max_element(x.train_distance.begin(),x.train_distance.end()));
    x.selection_seconds.push_back(elapsed+Seconds(start));
    if((c+1)%64==0 || c+1==m) {
      SaveSelection(x,path);
      std::cerr<<"selected "<<c+1<<'/'<<m<<" radius="<<x.cover_radius.back()
               <<" cumulative_seconds="<<x.selection_seconds.back()<<std::endl;
    }
  }
}
TQ8::EncodedSet Quantize(TQ8& model,const Set& docs,const Args& a) {
  const auto cache=Get(a,"cache");
  if(!fs::path(cache).parent_path().empty()) fs::create_directories(fs::path(cache).parent_path());
  Provenance(cache+".source",Identity(Get(a,"docs"))+"\nTQ8 corrected stock v1\n");
  TQ8::EncodedSet db;const auto t=Clock::now();
  if(fs::exists(cache)) {
    std::ifstream f(cache,std::ios::binary);model.load(f);db.load(f);
    if(!f || db.num_clouds()!=docs.size()) throw std::runtime_error("invalid TQ8 cache");
    std::cerr<<"loaded TQ8 cache seconds="<<Seconds(t)<<std::endl;
  } else {
    model.train(docs);db=model.encode(docs);
    std::cerr<<"encoded global TQ8 seconds="<<Seconds(t)<<" bytes="<<db.num_bytes()<<std::endl;
    std::ofstream f(cache+".tmp",std::ios::binary);model.save(f);db.save(f);f.close();
    if(!f) throw std::runtime_error("TQ8 cache write failed");fs::rename(cache+".tmp",cache);
  }
  return db;
}
std::vector<Neighbor> Exact(const Set& docs,const mvsic::ChamferIP_Point& q,uint32_t b) {
  auto scores=mvsic::OneToMany<mvsic::ChamferIP_Point,Set>::AllDistances(q,docs);
  std::vector<Neighbor> ranked(docs.size());
  parlay::parallel_for(0,docs.size(),[&](size_t i){ranked[i]={uint32_t(i),scores[i]};});
  Top(ranked,b);return ranked;
}
std::vector<Neighbor> Approx(const Set& docs,const mvsic::ChamferIP_Point& q,
    const TQ8& model,const TQ8::EncodedSet& db,uint32_t shortlist) {
  const auto query=model.quantize_query(q);
  std::vector<Neighbor> ranked(docs.size());
  parlay::parallel_for(0,docs.size(),[&](size_t i){ranked[i]={uint32_t(i),
      mvsic::turboquant_8bit_mv::turboquant_8bit_mv_chamfer_distance<false>(query,db,i)};});
  Top(ranked,shortlist);return ranked;
}
void Refine(std::vector<Neighbor>& ranked,const Set& docs,const mvsic::ChamferIP_Point& q,uint32_t b) {
  parlay::parallel_for(0,ranked.size(),[&](size_t i){ranked[i].distance=q.distance(docs[ranked[i].document]);});
  Top(ranked,b);
}
std::vector<std::vector<Neighbor>> Batch(const Set& train,const Set& docs,
    const Index& x,uint32_t start,uint32_t count,const TQ8& model,
    const TQ8::EncodedSet& db,uint32_t shortlist,uint32_t b) {
  std::vector<TQ8::EncodedQuery> queries;
  for(uint32_t j=0;j<count;++j) queries.push_back(model.quantize_query(train[x.centers[start+j]]));
  std::vector<const TQ8::EncodedQuery*> pointers;
  for(const auto& q:queries) pointers.push_back(&q);
  mvsic::turboquant_8bit_mv::FusedQueryBatch<false> fused; fused.Build(pointers);
  std::vector<std::pair<uint32_t,float>> scores(size_t(count)*docs.size());
  std::vector<std::vector<float>> workspaces(parlay::num_workers());
  mvsic::turboquant_8bit_mv::chamfer_score_all_fused(fused,db,scores.data(),&workspaces,true);
  std::vector<std::vector<Neighbor>> result(count);
  parlay::parallel_for(0,count,[&](size_t j) {
    auto& row=result[j]; row.resize(docs.size());
    for(size_t d=0;d<docs.size();++d) {const auto v=scores[j*docs.size()+d];row[d]={v.first,v.second};}
    Top(row,shortlist); Refine(row,docs,train[x.centers[start+j]],b);
  });
  return result;
}
double Overlap(const std::vector<Neighbor>& exact,const std::vector<Neighbor>& pool) {
  std::unordered_set<uint32_t> ids;for(auto x:pool) ids.insert(x.document);
  size_t n=0;for(auto x:exact) n+=ids.count(x.document);return double(n)/exact.size();
}
void Pilot(const Args& a) {
  Set train(Get(a,"train").c_str(),true),docs(Get(a,"docs").c_str(),true);
  const auto state=LoadSelection(Get(a,"state"));
  Provenance(Get(a,"state")+".source",Identity(Get(a,"train"))+"\nseed="+std::to_string(state.seed)+"\n");
  TQ8 model;auto db=Quantize(model,docs,a);
  const uint32_t b=std::stoul(Get(a,"b","512"));
  std::ofstream out(Get(a,"output"));
  out<<"train_row,kind,exact_seconds,tq8_seconds,shortlist,refine_seconds,pure_tq8_overlap,recovered_bucket_recall\n"<<std::setprecision(10);
  std::mt19937_64 rng(2026);
  std::vector<std::pair<uint32_t,std::string>> samples;
  for(size_t i=0;i<8;++i) samples.push_back({state.centers[i*(state.centers.size()-1)/7],"kcenter"});
  for(size_t i=0;i<8;++i) samples.push_back({std::uniform_int_distribution<uint32_t>(0,train.size()-1)(rng),"random_train"});
  for(auto [id,kind]:samples) {
    const auto q=train[id];auto t=Clock::now(); auto exact=Exact(docs,q,b);const double exact_s=Seconds(t);
    t=Clock::now();auto approx=Approx(docs,q,model,db,std::min<size_t>(b*4,docs.size()));const double tq_s=Seconds(t);
    auto pure=approx;Top(pure,b);const double pure_recall=Overlap(exact,pure);
    for(uint32_t c:{b*2,b*4}) {
      auto refined=approx;refined.resize(std::min<size_t>(c,refined.size()));
      t=Clock::now();Refine(refined,docs,q,b);const double refine_s=Seconds(t);
      out<<id<<','<<kind<<','<<exact_s<<','<<tq_s<<','<<c<<','<<refine_s<<','<<pure_recall<<','<<Overlap(exact,refined)<<'\n';
      out.flush();
      std::cerr<<"pilot row="<<id<<" shortlist="<<c<<" recovered="<<Overlap(exact,refined)
               <<" exact_s="<<exact_s<<" tq8_s="<<tq_s<<std::endl;
    }
  }
}
void Buckets(const Args& a) {
  const uint32_t batch=std::stoul(Get(a,"batch","1"));
  if(batch<1 || batch>32) throw std::invalid_argument("batch must be 1..32");
  auto x=LoadSelection(Get(a,"state")); const uint32_t m=std::stoul(Get(a,"m"));
  Provenance(Get(a,"state")+".source",Identity(Get(a,"train"))+"\nseed="+std::to_string(x.seed)+"\n");
  const uint32_t b=std::stoul(Get(a,"b","512")), shortlist=std::stoul(Get(a,"shortlist","1024"));
  const std::string mode=Get(a,"mode","tq8-refine"),output=Get(a,"output"),journal=Get(a,"journal");
  if(!m || m>x.centers.size() || !b || shortlist<b || (mode!="exact" && mode!="tq8-refine"))
    throw std::invalid_argument("invalid bucket parameters");
  Set train(Get(a,"train").c_str(),true),docs(Get(a,"docs").c_str(),true);
  if(train.size()!=x.train_size || train.get_dims()!=x.dimension || docs.get_dims()!=x.dimension || b>docs.size())
    throw std::runtime_error("bucket source dimensions mismatch");
  if(!fs::path(journal).parent_path().empty()) fs::create_directories(fs::path(journal).parent_path());
  Provenance(journal+".source",Identity(Get(a,"train"))+"\n"+Identity(Get(a,"docs"))+"\nseed="+
      std::to_string(x.seed)+"\nB="+std::to_string(b)+"\nmode="+mode+"\nshortlist="+std::to_string(shortlist)+"\n");
  x.corpus_size=docs.size();x.max_bucket_size=b;
  const size_t record_bytes=sizeof(uint32_t)*2+sizeof(double)+b*(sizeof(uint32_t)+sizeof(float));
  const size_t complete=fs::exists(journal)?fs::file_size(journal)/record_bytes:0;
  if(complete>x.centers.size()) throw std::runtime_error("journal exceeds selection state");
  if(fs::exists(journal) && fs::file_size(journal)%record_bytes) {
    // Discard only an incomplete final record of our own append-only journal.
    fs::resize_file(journal,complete*record_bytes);
    std::cerr<<"recovered incomplete journal tail"<<std::endl;
  }
  std::ifstream input(journal,std::ios::binary);
  for(size_t i=0;i<std::min<size_t>(complete,m);++i) {
    if(Read<uint32_t>(input)!=i || Read<uint32_t>(input)!=x.centers[i]) throw std::runtime_error("journal center mismatch");
    x.bucket_seconds.push_back(Read<double>(input));std::vector<Neighbor> bucket(b);
    for(auto& v:bucket) {v.document=Read<uint32_t>(input);v.distance=Read<float>(input);
      if(v.document>=docs.size() || !std::isfinite(v.distance)) throw std::runtime_error("invalid journal score");}
    x.buckets.push_back(std::move(bucket));
  }
  if(x.buckets.size()<m) {
    TQ8 model;TQ8::EncodedSet db;if(mode!="exact") db=Quantize(model,docs,a);
    const double elapsed=x.bucket_seconds.empty()?0:x.bucket_seconds.back();const auto start=Clock::now();
    std::ofstream append(journal,std::ios::binary|std::ios::app);
    for(uint32_t begin=x.buckets.size();begin<m;) {
      const uint32_t count=std::min(batch,m-begin);
      std::vector<std::vector<Neighbor>> batches;
      if(batch>1 && mode!="exact") batches=Batch(train,docs,x,begin,count,model,db,shortlist,b);
      for(uint32_t i=begin;i<begin+count;++i) {
      const auto q=train[x.centers[i]];std::vector<Neighbor> bucket;
      if(mode=="exact") bucket=Exact(docs,q,b);
      else if(batch>1) bucket=std::move(batches[i-begin]);
      else {bucket=Approx(docs,q,model,db,shortlist);Refine(bucket,docs,q,b);}
      x.bucket_seconds.push_back(elapsed+Seconds(start));x.buckets.push_back(std::move(bucket));
      Put(append,i);Put(append,x.centers[i]);Put(append,x.bucket_seconds.back());
      for(auto v:x.buckets.back()) {Put(append,v.document);Put(append,v.distance);}
      append.flush();if(!append) throw std::runtime_error("journal flush failed");
      if((i+1)%64==0 || i+1==m) std::cerr<<"built "<<i+1<<'/'<<m
          <<" cumulative_seconds="<<x.bucket_seconds.back()<<std::endl;
      }
      begin+=count;
    }
  }
  x.centers.resize(m);x.selection_seconds.resize(m);x.cover_radius.resize(m);
  emdivf::Save(x,output);std::cerr<<"saved "<<output<<std::endl;
}
int main(int argc,char** argv) {
  try {
    if(argc<2) throw std::invalid_argument("commands: select, pilot, buckets");
    Args a;for(int i=2;i<argc;i+=2) {
      if(i+1>=argc || std::string(argv[i]).rfind("--",0)!=0) throw std::invalid_argument("expected --key value");
      a[std::string(argv[i]).substr(2)]=argv[i+1];
    }
    const std::string command=argv[1];
    if(command=="select") Select(a);else if(command=="pilot") Pilot(a);
    else if(command=="buckets") Buckets(a);else throw std::invalid_argument("unknown command");
  } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}

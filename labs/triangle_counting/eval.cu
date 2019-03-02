/*! \brief File containing evaluation and grading code

This file contains evaluation and grading code.
This code is in a separate file so that a known-good version can be used for
automatic grading and students should not need to modify it.
*/

#include "helper.hpp"
#include "template.hu"

#include "pangolin/pangolin.hpp"

#ifndef GRAPH_PREFIX_PATH
#error "define GRAPH_PREFIX_PATH"
#endif

namespace gpu_algorithms_labs_evaluation {

enum Mode { LINEAR = 1, OTHER = 2 };

static int eval(const std::string &path, const Mode &mode) {

  assert(endsWith(path, ".bel") && "graph should be a .bel file");

  pangolin::logger::set_level(pangolin::logger::Level::TRACE);

  timer_start("Reading graph data");
  // create a reader for .bel files
  pangolin::BELReader reader(path);
  // read all edges from the file
  pangolin::EdgeList edgeList = reader.read_all();
  timer_stop(); // reading graph data

  timer_start("building unified memory CSR");
  // build a csr/coo matrix from the edge list
  pangolin::COO<uint32_t> coo = pangolin::COO<uint32_t>::from_edgelist(
    edgeList,
    [](const pangolin::Edge &e) {return e.first <= e.second; } // keep src > dst
  );
  timer_stop(); // building unified memory CSR

  timer_start("counting triangles");
  uint64_t actual = count_triangles(coo.view(), mode);
  timer_stop(); // counting triangles

  timer_start("comparing result");
  pangolin::Config c;
  c.gpus_ = {0};
  c.type_ = "impact2019";
  c.storage_ = "um";
  c.kernel_ = "linear";

  auto tc = pangolin::TriangleCounter::CreateTriangleCounter(c);
  tc->read_data(path);
  tc->setup_data();
  uint64_t expected = tc->count();
  timer_stop();

  INFO("verifying with graph " << path);
  REQUIRE(actual == expected);

  return 0;
}
  

TEST_CASE("graph500-scale18-ef16_adj", "") {
  SECTION("LINEAR") { eval(GRAPH_PREFIX_PATH "/graph500-scale18-ef16_adj.bel", LINEAR); }
  SECTION("OTHER") { eval(GRAPH_PREFIX_PATH "/graph500-scale18-ef16_adj.bel", OTHER); }
}

TEST_CASE("amazon0302_adj", "") {
  SECTION("LINEAR") { eval(GRAPH_PREFIX_PATH "/amazon0302_adj.bel", LINEAR); }
  SECTION("OTHER") { eval(GRAPH_PREFIX_PATH "/amazon0302_adj.bel", OTHER); }
}

TEST_CASE("roadNet-CA_adj", "") {
  SECTION("LINEAR") { eval(GRAPH_PREFIX_PATH "/roadNet-CA_adj.bel", LINEAR); }
  SECTION("OTHER") { eval(GRAPH_PREFIX_PATH "/roadNet-CA_adj.bel", OTHER); }
}


} // namespace gpu_algorithms_labs_evaluation

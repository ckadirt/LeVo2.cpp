#include "levo-progress.h"

#include <cassert>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>

namespace {

template <typename Function>
bool throws(Function && function) {
    try { function(); } catch (const std::exception &) { return true; }
    return false;
}

} // namespace

int main() {
    using namespace levo_cli;
    assert(parse_progress_mode("plain") == progress_mode::plain);
    assert(parse_progress_mode("json") == progress_mode::json);
    assert(parse_progress_mode("none") == progress_mode::none);
    assert(throws([] { (void) parse_progress_mode("verbose"); }));
    assert(parse_progress_interval("0") == 0.0);
    assert(parse_progress_interval("1.25") == 1.25);
    assert(throws([] { (void) parse_progress_interval("-1"); }));
    assert(throws([] { (void) parse_progress_interval("nan"); }));

    levo::generation_progress generation;
    generation.stage = levo::generation_stage::generating;
    generation.completed_steps = 25;
    generation.total_steps = 100;
    generation.requested_frames = 50;
    generation.elapsed_seconds = 7.0;
    generation.stage_elapsed_seconds = 5.0;
    const std::string generation_event = generation_json(generation);
    assert(generation_event.front() == '{' && generation_event.back() == '}');
    assert(generation_event.find("\"stage\":\"generating\"") != std::string::npos);
    assert(generation_event.find("\"eta_seconds\":15") != std::string::npos);

    levo::render_progress render;
    render.stage = levo::render_stage::generating_latents;
    render.completed_windows = 1;
    render.total_windows = 3;
    render.current_window = 2;
    render.completed_steps = 10;
    render.total_steps = 50;
    render.elapsed_seconds = 30.0;
    render.stage_elapsed_seconds = 20.0;
    const std::string render_event = render_json(render);
    assert(render_event.front() == '{' && render_event.back() == '}');
    assert(render_event.find("\"current_window\":2") != std::string::npos);
    assert(render_plain(render).find("window 2/3, Euler 10/50") != std::string::npos);

    std::ostringstream json_output;
    {
        generation_progress_writer writer(progress_mode::json, 0.0, json_output);
        writer.update(generation);
        generation.stage = levo::generation_stage::complete;
        writer.update(generation);
    }
    assert(json_output.str().find("generation_progress") != std::string::npos);
    assert(json_output.str().find("\"stage\":\"complete\"") != std::string::npos);

    std::ostringstream quiet_output;
    {
        render_progress_writer writer(progress_mode::none, 0.0, quiet_output);
        writer.update(render);
    }
    assert(quiet_output.str().empty());

    std::cout << "progress formatting ok\n";
    return 0;
}

#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <bingocpp/agraph/agraph.h>
#include <bingocpp/agraph/operator_definitions.h>
#include <bingocpp/agraph/string_generation.h>
#include <bingocpp/agraph/evaluation_backend/evaluation_backend.h>
#include <bingocpp/agraph/simplification_backend/simplification_backend.h>
#include <bingocpp/agraph/constants.h>

namespace bingo
{

  namespace
  {

    constexpr int kOpIdx = 0;     // Operation index
    constexpr int kFirstArgumentIndex = 1; // First parameter index
    constexpr int kSecondArgumentIndex = 2; // Second parameter index
    constexpr int kInitialCommandRows = 0;
    constexpr int kInitialCommandCols = 3;
    constexpr int kInitialConstantsCol = 1;
    const double kFitnessNotSet = 1e9;

    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

  } // namespace

  AGraph::AGraph(const bool use_simplification)
  {
    command_array_ = Eigen::ArrayX3i(kInitialCommandRows, kInitialCommandCols);
    simplified_command_array_ = Eigen::ArrayX3i(kInitialCommandRows, kInitialCommandCols);
    simplified_constants_ = Eigen::ArrayXXd(kInitialCommandRows, kInitialConstantsCol);
    needs_opt_ = false;
    fitness_ = kFitnessNotSet;
    fit_set_ = false;
    genetic_age_ = 0;
    modified_ = false;
    use_simplification_ = use_simplification;
  }

  AGraph::AGraph(const AGraph &agraph)
  {
    command_array_ = agraph.command_array_;
    simplified_command_array_ = agraph.simplified_command_array_;
    simplified_constants_ = agraph.simplified_constants_;
    needs_opt_ = agraph.needs_opt_;
    fitness_ = agraph.fitness_;
    fit_set_ = agraph.fit_set_;
    genetic_age_ = agraph.genetic_age_;
    modified_ = agraph.modified_;
    use_simplification_ = agraph.use_simplification_;
  }

  AGraph::AGraph(const AGraphState &state)
  {
    command_array_ = std::get<0>(state);
    simplified_command_array_ = std::get<1>(state);
    simplified_constants_ = std::get<2>(state);
    needs_opt_ = std::get<3>(state);
    fitness_ = std::get<4>(state);
    fit_set_ = std::get<5>(state);
    genetic_age_ = std::get<6>(state);
    modified_ = std::get<7>(state);
    use_simplification_ = std::get<8>(state);
  }

  AGraph AGraph::Copy()
  {
    return AGraph(*this);
  }

  AGraphState AGraph::DumpState()
  {
    return AGraphState(command_array_, simplified_command_array_,
                       simplified_constants_, needs_opt_, fitness_, fit_set_,
                       genetic_age_, modified_, use_simplification_);
  }

  const Eigen::ArrayX3i &AGraph::GetCommandArray() const
  {
    return command_array_;
  }

  Eigen::ArrayX3i &AGraph::GetCommandArrayModifiable()
  {
    notify_agraph_modification();
    return command_array_;
  }

  void AGraph::SetCommandArray(const Eigen::ArrayX3i &command_array)
  {
    command_array_ = command_array;
    notify_agraph_modification();
  }

  void AGraph::notify_agraph_modification()
  {
    fitness_ = kFitnessNotSet;
    fit_set_ = false;
    modified_ = true;
  }

  double AGraph::GetFitness() const
  {
    return fitness_;
  }

  void AGraph::SetFitness(double fitness)
  {
    fitness_ = fitness;
    fit_set_ = true;
  }

  bool AGraph::IsFitnessSet() const
  {
    return fit_set_;
  }

  void AGraph::SetFitnessStatus(bool val)
  {
    fit_set_ = val;
  }

  void AGraph::SetGeneticAge(const int age)
  {
    genetic_age_ = age;
  }

  int AGraph::GetGeneticAge() const
  {
    return genetic_age_;
  }

  std::vector<bool> AGraph::GetUtilizedCommands() const
  {
    return simplification_backend::GetUtilizedCommands(command_array_);
  }

  bool AGraph::NeedsLocalOptimization()
  {
    if (modified_)
    {
      update();
    }
    return needs_opt_;
  }

  int AGraph::GetNumberLocalOptimizationParams()
  {
    if (modified_)
    {
      update();
    }
    return simplified_constants_.rows();
  }

  void AGraph::SetLocalOptimizationParams(Eigen::Ref<Eigen::ArrayXXd> params)
  {
    simplified_constants_ = params;
    needs_opt_ = false;
  }

  void AGraph::SetLocalOptimizationParamsV(Eigen::VectorXd params)
  {
    simplified_constants_ = params;
    needs_opt_ = false;
  }

  void AGraph::SetLocalOptimizationParamsA(Eigen::ArrayXXd params)
  {
    simplified_constants_ = params;
    needs_opt_ = false;
  }

  const Eigen::ArrayXXd &AGraph::GetLocalOptimizationParams() const
  {
    return simplified_constants_;
  }

  Eigen::ArrayXXd
  AGraph::EvaluateEquationAt(const Eigen::ArrayXXd &x)
  {
    if (modified_)
    {
      update();
    }
    Eigen::ArrayXXd f_of_x;
    try
    {
      f_of_x = evaluation_backend::Evaluate(this->simplified_command_array_,
                                            x,
                                            this->simplified_constants_);
      return f_of_x;
    }
    catch (const std::underflow_error &ue)
    {
      return Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
    }
    catch (const std::overflow_error &oe)
    {
      return Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
    }
  }

  EvalAndDerivative
  AGraph::EvaluateEquationWithXGradientAt(const Eigen::ArrayXXd &x)
  {
    if (modified_)
    {
      update();
    }
    EvalAndDerivative df_dx;
    try
    {
      df_dx = evaluation_backend::EvaluateWithDerivative(this->simplified_command_array_,
                                                         x,
                                                         this->simplified_constants_,
                                                         true);
      return df_dx;
    }
    catch (const std::underflow_error &ue)
    {
      Eigen::ArrayXXd nan_array =
          Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
      return std::make_pair(nan_array, nan_array);
    }
    catch (const std::overflow_error &oe)
    {
      Eigen::ArrayXXd nan_array =
          Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
      return std::make_pair(nan_array, nan_array);
    }
  }

  EvalAndDerivative
  AGraph::EvaluateEquationWithLocalOptGradientAt(const Eigen::ArrayXXd &x)
  {
    if (modified_)
    {
      update();
    }
    EvalAndDerivative df_dc;
    try
    {
      df_dc = evaluation_backend::EvaluateWithDerivative(this->simplified_command_array_,
                                                         x,
                                                         this->simplified_constants_,
                                                         false);
      return df_dc;
    }
    catch (const std::underflow_error &ue)
    {
      Eigen::ArrayXXd nan_array =
          Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
      return std::make_pair(nan_array, nan_array);
    }
    catch (const std::overflow_error &oe)
    {
      Eigen::ArrayXXd nan_array =
          Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
      return std::make_pair(nan_array, nan_array);
    }
  }

  std::ostream &operator<<(std::ostream &strm, AGraph &graph)
  {
    return strm << graph.GetConsoleString();
  }

  std::string AGraph::GetConsoleString()
  {
    return AGraph::GetFormattedString("console", false);
  }

  std::string AGraph::GetFormattedString(std::string format, bool raw)
  {
    if (raw)
    {
      return string_generation::GetFormattedString(format, this->command_array_, Eigen::VectorXd(0));
    }
    if (modified_)
    {
      update();
    }
    return string_generation::GetFormattedString(format, this->simplified_command_array_, this->simplified_constants_);
  }

  int AGraph::GetComplexity()
  {
    if (modified_)
    {
      update();
    }
    return simplified_command_array_.rows();
  }

  int AGraph::Distance(const AGraph &agraph)
  {
    return (command_array_ != agraph.GetCommandArray()).count();
  }

  void AGraph::update() {
    updateSimplifiedCommandArray();
    updateConstantsArray();
    modified_ = false;
}

void AGraph::updateSimplifiedCommandArray() {
    if (use_simplification_) {
        simplified_command_array_ = simplification_backend::PythonSimplifyStack(command_array_);
    } else {
        simplified_command_array_ = simplification_backend::SimplifyStack(command_array_);
    }
}

void AGraph::updateConstantsArray() {
    int new_const_number = countAndUpdateConstants();
    resizeConstantsArrayIfNeeded(new_const_number);
}

int AGraph::countAndUpdateConstants() {
    int new_const_number = 0;
    for (int i = 0; i < simplified_command_array_.rows(); i++) {
        if (simplified_command_array_(i, kOpIdx) == Op::kConstant) {
            simplified_command_array_.row(i) << Op::kConstant, new_const_number, new_const_number;
            new_const_number++;
        }
    }
    return new_const_number;
}

void AGraph::resizeConstantsArrayIfNeeded(int const_number_input){
  int optimization_aggression = 0; 
  
  if (optimization_aggression == 0 && const_number_input <= simplified_constants_.rows()){
    simplified_constants_.conservativeResize(const_number_input, Eigen::NoChange);
    return; 
  }

  if (optimization_aggression == 1 && const_number_input == simplified_constants_.rows()){
    // reuse old constants
    return; 
  }

  performDefaultConstantResize(const_number_input);
}

void AGraph::performDefaultConstantResize(int const_number_input){
  simplified_constants_.resize(const_number_input, 1);
  simplified_constants_.setOnes(const_number_input, 1);
  if (const_number_input > 0){
    needs_opt_ = true;
  }
}

} // namespace bingo

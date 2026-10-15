"""FeatureForge: automated, leak-free feature engineering for tabular data.

Two ways in:

- ``build_features(train, test, target, tables=...)`` runs the whole pipeline (related tables,
  structure-triggered blocks, forecasting check, gated search) and writes parquet files. The
  ``featureforge`` command line is the same function.
- ``FeatureForge`` is the search on one table::

      forge = FeatureForge(time_budget=300).fit(X_train, y_train, X_unlabeled=X_test)
      X_train_fe = forge.transform_train(X_train)   # out-of-fold target-dependent features
      X_test_fe = forge.transform(X_test)
"""
from featureforge.blocks.forecast import ForecastFeatures
from featureforge.blocks.relational import Child, RelatedTables
from featureforge.forge import FeatureForge
from featureforge.pipeline import PipelineConfig, build_features

__version__ = "1.0.0"

__all__ = ["FeatureForge", "ForecastFeatures", "Child", "RelatedTables", "PipelineConfig", "build_features",
           "__version__"]

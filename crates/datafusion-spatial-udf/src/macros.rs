//! Boilerplate for the many one-argument functions.
//!
//! Most accessors share the same shape: take one geometry, check it at plan time, call one kernel,
//! return a plain Arrow column. The macro below writes that shape so each function needs only its
//! name, its return type and its kernel.

/// Define a scalar UDF that takes one geometry and returns a plain Arrow column.
///
/// The generated `return_field_from_args` validates that the argument really is a geometry, so a
/// mistyped query fails at plan time rather than on the first batch.
#[macro_export]
macro_rules! unary_geometry_udf {
    (
        $(#[$meta:meta])*
        $struct:ident, $sql_name:literal, $postgis_name:literal, $return_type:expr, $kernel:path
    ) => {
        $(#[$meta])*
        #[derive(Debug, PartialEq, Eq, Hash)]
        pub struct $struct {
            signature: ::datafusion::logical_expr::Signature,
        }

        impl $struct {
            /// Build the UDF.
            pub fn new() -> Self {
                Self {
                    // A geometry argument. Its Arrow storage type varies by encoding, so the
                    // signature accepts anything and the return field does the real check.
                    signature: ::datafusion::logical_expr::Signature::any(
                        1,
                        ::datafusion::logical_expr::Volatility::Immutable,
                    ),
                }
            }
        }

        impl Default for $struct {
            fn default() -> Self {
                Self::new()
            }
        }

        impl ::datafusion::logical_expr::ScalarUDFImpl for $struct {
            // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
            #[cfg(feature = "df53")]
            fn as_any(&self) -> &dyn ::std::any::Any {
                self
            }

            fn name(&self) -> &str {
                $sql_name
            }

            fn signature(&self) -> &::datafusion::logical_expr::Signature {
                &self.signature
            }

            fn return_type(
                &self,
                _arg_types: &[::arrow_schema::DataType],
            ) -> ::datafusion::common::Result<::arrow_schema::DataType> {
                Ok($return_type)
            }

            fn return_field_from_args(
                &self,
                args: ::datafusion::logical_expr::ReturnFieldArgs,
            ) -> ::datafusion::common::Result<::arrow_schema::FieldRef> {
                $crate::util::geo_type($postgis_name, 0, &args.arg_fields[0])?;
                Ok(::std::sync::Arc::new(::arrow_schema::Field::new(
                    $sql_name,
                    $return_type,
                    true,
                )))
            }

            fn invoke_with_args(
                &self,
                args: ::datafusion::logical_expr::ScalarFunctionArgs,
            ) -> ::datafusion::common::Result<::datafusion::logical_expr::ColumnarValue> {
                let scalar_input = $crate::util::all_scalar(&args.args);
                let array = $crate::util::geo_array(&args.args[0], &args.arg_fields[0])?;
                let result = $kernel(array.as_ref()).map_err($crate::util::to_df)?;
                $crate::util::wrap_result(::std::sync::Arc::new(result), scalar_input)
            }
        }
    };
}

/// Define a scalar UDF that takes one geometry and returns a geometry.
///
/// `$output_type` maps the input [`GeoArrowType`][geoarrow_schema::GeoArrowType] to the output
/// one. The map runs at plan time. So the output field holds the right extension metadata
/// and the coordinate reference system survives the call.
#[macro_export]
macro_rules! unary_transform_udf {
    (
        $(#[$meta:meta])*
        $struct:ident, $sql_name:literal, $postgis_name:literal, $output_type:expr, $kernel:path
    ) => {
        $(#[$meta])*
        #[derive(Debug, PartialEq, Eq, Hash)]
        pub struct $struct {
            signature: ::datafusion::logical_expr::Signature,
        }

        impl $struct {
            /// Build the UDF.
            pub fn new() -> Self {
                Self {
                    signature: ::datafusion::logical_expr::Signature::any(
                        1,
                        ::datafusion::logical_expr::Volatility::Immutable,
                    ),
                }
            }

            fn output_for(
                input: &::geoarrow_schema::GeoArrowType,
            ) -> ::datafusion::common::Result<::geoarrow_schema::GeoArrowType> {
                let mapper: fn(
                    &::geoarrow_schema::GeoArrowType,
                )
                    -> ::geoarrow_schema::error::GeoArrowResult<::geoarrow_schema::GeoArrowType> =
                    $output_type;
                mapper(input).map_err($crate::util::to_df)
            }
        }

        impl Default for $struct {
            fn default() -> Self {
                Self::new()
            }
        }

        impl ::datafusion::logical_expr::ScalarUDFImpl for $struct {
            // DataFusion 54 dropped `as_any` from this trait, so it exists on df53 only.
            #[cfg(feature = "df53")]
            fn as_any(&self) -> &dyn ::std::any::Any {
                self
            }

            fn name(&self) -> &str {
                $sql_name
            }

            fn signature(&self) -> &::datafusion::logical_expr::Signature {
                &self.signature
            }

            fn return_type(
                &self,
                _arg_types: &[::arrow_schema::DataType],
            ) -> ::datafusion::common::Result<::arrow_schema::DataType> {
                ::datafusion::common::plan_err!(
                    "{} needs the argument field to determine its return type",
                    $postgis_name
                )
            }

            fn return_field_from_args(
                &self,
                args: ::datafusion::logical_expr::ReturnFieldArgs,
            ) -> ::datafusion::common::Result<::arrow_schema::FieldRef> {
                let input = $crate::util::geo_type($postgis_name, 0, &args.arg_fields[0])?;
                let output = Self::output_for(&input)?;
                Ok($crate::util::geo_field($sql_name, &output))
            }

            fn invoke_with_args(
                &self,
                args: ::datafusion::logical_expr::ScalarFunctionArgs,
            ) -> ::datafusion::common::Result<::datafusion::logical_expr::ColumnarValue> {
                let scalar_input = $crate::util::all_scalar(&args.args);
                let array = $crate::util::geo_array(&args.args[0], &args.arg_fields[0])?;
                let result = $kernel(array.as_ref()).map_err($crate::util::to_df)?;
                $crate::util::wrap_geo_result(result, scalar_input)
            }
        }
    };
}

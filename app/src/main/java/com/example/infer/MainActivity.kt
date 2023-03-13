package com.example.infer

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import java.nio.FloatBuffer

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize the views
        val cgpa = findViewById<EditText>( R.id.cgpa ).text
        val iq = findViewById<EditText>( R.id.iq ).text
        val profile_score = findViewById<EditText>( R.id.profile_score ).text
        val outputTextView = findViewById<TextView>( R.id.output_textview )
        val button = findViewById<Button>( R.id.predict_button )

        button.setOnClickListener {
            // Parse input from inputEditText
            val icgpa: Float = cgpa.toString().toFloat()
            val iiq: Float = iq.toString().toFloat()
            val ips: Float = profile_score.toString().toFloat()
            val inputs: FloatArray = Array<Float>(3){0f}.toFloatArray()
            inputs.set(0, icgpa)
            inputs.set(1, iiq)
            inputs.set(2, ips)
            if ( inputs != null ) {
                val ortEnvironment = OrtEnvironment.getEnvironment()
                val ortSession = createORTSession( ortEnvironment )
                val output = runPrediction( inputs , ortSession , ortEnvironment )
                outputTextView.text = "Output is ${output}"
            }
            else {
                Toast.makeText( this , "Please check the inputs" , Toast.LENGTH_LONG ).show()
            }
        }

    }

    // Create an OrtSession
    private fun createORTSession( ortEnvironment: OrtEnvironment ) : OrtSession {
        val modelBytes = resources.openRawResource( R.raw.sklearn_model ).readBytes()
        return ortEnvironment.createSession( modelBytes )
    }

    // Make predictions with given inputs
    private fun runPrediction( input : FloatArray , ortSession: OrtSession , ortEnvironment: OrtEnvironment ) : Long {
        // Get the name of the input node
        val inputName = ortSession.inputNames?.iterator()?.next()
        // Make a FloatBuffer of the inputs
        val floatBufferInputs = FloatBuffer.wrap( input )
        // Create input tensor with floatBufferInputs of shape ( 1 , 1 )
        val inputTensor = OnnxTensor.createTensor( ortEnvironment , floatBufferInputs , longArrayOf( 1, 3 ) )
        // Run the model
        val results = ortSession.run( mapOf( inputName to inputTensor ) )
        //results.forEach {println(">>>>> ${it} ")}

        // Fetch and return the results
        val output = results[0].value as LongArray
        return output[0]
    }


}
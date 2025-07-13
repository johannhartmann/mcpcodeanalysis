<?php
/**
 * Sample PHP file for testing parser functionality.
 * This demonstrates various PHP language features.
 */

namespace Sample\Testing;

use DateTime;
use Exception;
use Sample\Base\AbstractClass;
use Sample\Utils\Helper as UtilHelper;

/**
 * Sample interface for testing
 */
interface SampleInterface {
    public function process(string $data): bool;
}

/**
 * Sample trait for testing
 */
trait LoggerTrait {
    /**
     * Log a message
     *
     * @param string $message The message to log
     * @return void
     */
    public function log(string $message): void {
        echo "[LOG] " . $message . PHP_EOL;
    }
}

/**
 * A sample abstract class for testing inheritance
 */
abstract class BaseClass {
    protected string $name;

    abstract public function getName(): string;
}

/**
 * Main sample class demonstrating various PHP features
 *
 * @package Sample\Testing
 * @author Test Author
 */
class SampleClass extends BaseClass implements SampleInterface {
    use LoggerTrait;

    private DateTime $createdAt;
    private static int $instanceCount = 0;

    /**
     * Constructor for SampleClass
     *
     * @param string $name The name to set
     * @param DateTime|null $date Optional date parameter
     */
    public function __construct(string $name, ?DateTime $date = null) {
        $this->name = $name;
        $this->createdAt = $date ?? new DateTime();
        self::$instanceCount++;
    }

    /**
     * Get the name property
     *
     * @return string
     */
    public function getName(): string {
        return $this->name;
    }

    /**
     * Process data according to interface contract
     *
     * @param string $data Data to process
     * @return bool Success status
     * @throws Exception When processing fails
     */
    public function process(string $data): bool {
        $this->log("Processing: " . $data);

        if (empty($data)) {
            throw new Exception("Data cannot be empty");
        }

        return true;
    }

    /**
     * Static method to get instance count
     *
     * @return int Number of instances created
     */
    public static function getInstanceCount(): int {
        return self::$instanceCount;
    }

    /**
     * Magic method for string representation
     *
     * @return string
     */
    public function __toString(): string {
        return sprintf("SampleClass[name=%s]", $this->name);
    }

    /**
     * Generator function example
     *
     * @param int $max Maximum number to generate
     * @return iterable
     */
    public function generateNumbers(int $max): iterable {
        for ($i = 0; $i <= $max; $i++) {
            yield $i;
        }
    }
}

/**
 * Standalone function for testing
 *
 * @param array $items Array of items to process
 * @param callable $callback Callback to apply
 * @return array Processed items
 */
function processItems(array $items, callable $callback): array {
    return array_map($callback, $items);
}

/**
 * Another standalone function with default parameter
 *
 * @param string $prefix Prefix to add
 * @param string $suffix Suffix to add
 * @return string Combined string
 */
function formatString(string $prefix = "START", string $suffix = "END"): string {
    return $prefix . " - content - " . $suffix;
}

// Example of closure
$multiplier = function(int $x, int $y): int {
    return $x * $y;
};

// Arrow function (PHP 7.4+)
$squared = fn($n) => $n * $n;

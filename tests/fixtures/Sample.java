/**
 * Sample Java file for testing parser functionality.
 * This demonstrates various Java language features.
 */
package com.example.testing;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.time.LocalDateTime;
import java.io.*;

/**
 * Sample interface for testing
 */
public interface Processor<T> {
    /**
     * Process an item
     * @param item The item to process
     * @return Processed result
     */
    T process(T item);
}

/**
 * Sample abstract class demonstrating inheritance
 */
abstract class AbstractService {
    protected String serviceName;

    public AbstractService(String name) {
        this.serviceName = name;
    }

    /**
     * Abstract method to be implemented by subclasses
     * @return Service status
     */
    public abstract boolean isRunning();
}

/**
 * Main sample class demonstrating various Java features
 *
 * @author Test Author
 * @version 1.0
 * @since 2024
 */
@SuppressWarnings("unused")
public class Sample extends AbstractService implements Processor<String>, Serializable {
    private static final long serialVersionUID = 1L;
    private static int instanceCount = 0;

    private final Long id;
    private String name;
    private LocalDateTime createdAt;
    private List<String> tags;

    /**
     * Default constructor
     */
    public Sample() {
        this(null, "default");
    }

    /**
     * Constructor with parameters
     *
     * @param id The unique identifier
     * @param name The name of the sample
     */
    public Sample(Long id, String name) {
        super("SampleService");
        this.id = id;
        this.name = name;
        this.createdAt = LocalDateTime.now();
        this.tags = new ArrayList<>();
        instanceCount++;
    }

    /**
     * Get the ID
     * @return The ID value
     */
    public Long getId() {
        return id;
    }

    /**
     * Get the name
     * @return The name value
     */
    public String getName() {
        return name;
    }

    /**
     * Set the name
     * @param name The new name
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Process a string according to interface contract
     *
     * @param item The string to process
     * @return Processed string
     * @throws IllegalArgumentException if item is null
     */
    @Override
    public String process(String item) {
        if (item == null) {
            throw new IllegalArgumentException("Item cannot be null");
        }
        return item.toUpperCase();
    }

    /**
     * Check if service is running
     * @return Always returns true for this sample
     */
    @Override
    public boolean isRunning() {
        return true;
    }

    /**
     * Static method to get instance count
     * @return Number of instances created
     */
    public static int getInstanceCount() {
        return instanceCount;
    }

    /**
     * Generic method example
     * @param <T> Type parameter
     * @param list List to process
     * @return First element or null
     */
    public <T> T getFirst(List<T> list) {
        return list.isEmpty() ? null : list.get(0);
    }

    /**
     * Method with varargs
     * @param values Variable number of strings
     * @return Concatenated string
     */
    public String concatenate(String... values) {
        return String.join(", ", values);
    }

    /**
     * Method demonstrating Java 8 features
     * @param items List of items to filter
     * @param prefix Prefix to match
     * @return Filtered list
     */
    public List<String> filterByPrefix(List<String> items, String prefix) {
        return items.stream()
            .filter(item -> item.startsWith(prefix))
            .map(String::toLowerCase)
            .collect(Collectors.toList());
    }

    /**
     * Method returning Optional
     * @param index Index to retrieve
     * @return Optional containing the tag or empty
     */
    public Optional<String> getTag(int index) {
        if (index >= 0 && index < tags.size()) {
            return Optional.of(tags.get(index));
        }
        return Optional.empty();
    }

    /**
     * toString implementation
     * @return String representation
     */
    @Override
    public String toString() {
        return String.format("Sample[id=%d, name=%s]", id, name);
    }

    /**
     * Inner class example
     */
    public static class Builder {
        private Long id;
        private String name;

        public Builder withId(Long id) {
            this.id = id;
            return this;
        }

        public Builder withName(String name) {
            this.name = name;
            return this;
        }

        public Sample build() {
            return new Sample(id, name);
        }
    }

    /**
     * Enum example
     */
    public enum Status {
        ACTIVE("Active"),
        INACTIVE("Inactive"),
        PENDING("Pending");

        private final String displayName;

        Status(String displayName) {
            this.displayName = displayName;
        }

        public String getDisplayName() {
            return displayName;
        }
    }
}

/**
 * Another class in the same file
 */
@Deprecated
class Helper {
    /**
     * Utility method
     * @param value Value to check
     * @return true if valid
     */
    public static boolean isValid(String value) {
        return value != null && !value.trim().isEmpty();
    }
}
